#include <flexiv/rdk/gripper.hpp>
#include <flexiv/rdk/robot.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr std::int64_t kRingMagic = 0x4446425252494e47LL & 0x7fffffffffffffffLL;
constexpr std::int64_t kRingVersion = 1;
constexpr std::size_t kHeaderLength = 8;
constexpr std::size_t kWriteCountIndex = 5;
constexpr std::int64_t kFloat64Code = 1;
constexpr int kStopCommand = 2;
constexpr std::chrono::nanoseconds kLoopPeriod{1'000'000};

std::atomic<bool> g_stop{false};

void SignalHandler(int)
{
    g_stop.store(true, std::memory_order_relaxed);
}

std::int64_t MonotonicNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

class Ring
{
public:
    explicit Ring(std::string name)
    : name_(std::move(name))
    {
        std::string posix_name = name_;
        if (posix_name.empty() || posix_name.front() != '/') {
            posix_name.insert(posix_name.begin(), '/');
        }
        fd_ = shm_open(posix_name.c_str(), O_RDWR, 0);
        if (fd_ < 0) {
            throw std::runtime_error("shm_open(" + posix_name + "): " + std::strerror(errno));
        }
        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            throw std::runtime_error("fstat(" + posix_name + "): " + std::strerror(errno));
        }
        size_ = static_cast<std::size_t>(st.st_size);
        mapping_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            mapping_ = nullptr;
            throw std::runtime_error("mmap(" + posix_name + "): " + std::strerror(errno));
        }
        header_ = static_cast<std::int64_t*>(mapping_);
        if (header_[0] != kRingMagic || header_[1] != kRingVersion) {
            throw std::runtime_error(name_ + " is not a DFC ring v1");
        }
        capacity_ = static_cast<std::size_t>(header_[2]);
        dim_ = static_cast<std::size_t>(header_[3]);
        if (header_[4] != kFloat64Code) {
            throw std::runtime_error(name_ + " must have float64 payloads");
        }
        seq_ = header_ + kHeaderLength;
        timestamps_ = seq_ + capacity_;
        data_ = reinterpret_cast<double*>(timestamps_ + capacity_);
        const auto required = (kHeaderLength + 2 * capacity_) * sizeof(std::int64_t)
            + capacity_ * dim_ * sizeof(double);
        if (required > size_) {
            throw std::runtime_error(name_ + " has a truncated payload");
        }
    }

    Ring(const Ring&) = delete;
    Ring& operator=(const Ring&) = delete;

    ~Ring()
    {
        if (mapping_ != nullptr) {
            munmap(mapping_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    std::size_t dim() const { return dim_; }
    std::size_t capacity() const { return capacity_; }

    std::int64_t write_count() const
    {
        return std::atomic_ref<std::int64_t>(header_[kWriteCountIndex])
            .load(std::memory_order_acquire);
    }

    bool ReadAt(std::int64_t index, std::vector<double>& out, std::int64_t* timestamp = nullptr) const
    {
        if (index < 0) {
            return false;
        }
        const auto slot = static_cast<std::size_t>(index) % capacity_;
        const auto first = std::atomic_ref<std::int64_t>(seq_[slot])
                               .load(std::memory_order_acquire);
        if (first != index) {
            return false;
        }
        out.resize(dim_);
        std::copy_n(data_ + slot * dim_, dim_, out.begin());
        const auto ts = timestamps_[slot];
        const auto second = std::atomic_ref<std::int64_t>(seq_[slot])
                                .load(std::memory_order_acquire);
        if (second != index) {
            return false;
        }
        if (timestamp != nullptr) {
            *timestamp = ts;
        }
        return true;
    }

    bool Latest(std::vector<double>& out, std::int64_t& timestamp) const
    {
        for (int retry = 0; retry < 4; ++retry) {
            const auto count = write_count();
            if (count <= 0) {
                return false;
            }
            if (ReadAt(count - 1, out, &timestamp)) {
                return true;
            }
        }
        return false;
    }

    void Append(const double* values, std::size_t count, std::int64_t timestamp)
    {
        if (count != dim_) {
            throw std::runtime_error(name_ + " write dimension mismatch");
        }
        auto write_count_ref = std::atomic_ref<std::int64_t>(header_[kWriteCountIndex]);
        const auto index = write_count_ref.load(std::memory_order_relaxed);
        const auto slot = static_cast<std::size_t>(index) % capacity_;
        auto seq_ref = std::atomic_ref<std::int64_t>(seq_[slot]);
        seq_ref.store(-1, std::memory_order_release);
        std::copy_n(values, count, data_ + slot * dim_);
        timestamps_[slot] = timestamp;
        seq_ref.store(index, std::memory_order_release);
        write_count_ref.store(index + 1, std::memory_order_release);
    }

    template <typename Container>
    void Append(const Container& values, std::int64_t timestamp)
    {
        Append(values.data(), values.size(), timestamp);
    }

private:
    std::string name_;
    int fd_{-1};
    std::size_t size_{0};
    void* mapping_{nullptr};
    std::int64_t* header_{nullptr};
    std::int64_t* seq_{nullptr};
    std::int64_t* timestamps_{nullptr};
    double* data_{nullptr};
    std::size_t capacity_{0};
    std::size_t dim_{0};
};

struct Args
{
    std::map<std::string, std::string> values;

    explicit Args(int argc, char** argv)
    {
        for (int i = 1; i < argc; ++i) {
            const std::string key(argv[i]);
            if (!key.starts_with("--")) {
                throw std::invalid_argument("unexpected positional argument: " + key);
            }
            if (i + 1 >= argc || std::string(argv[i + 1]).starts_with("--")) {
                values[key] = "1";
            } else {
                values[key] = argv[++i];
            }
        }
    }

    bool has(const std::string& key) const { return values.contains(key); }

    std::string get(const std::string& key) const
    {
        const auto it = values.find(key);
        if (it == values.end()) {
            throw std::invalid_argument("missing required argument " + key);
        }
        return it->second;
    }

    std::string get(const std::string& key, const std::string& fallback) const
    {
        const auto it = values.find(key);
        return it == values.end() ? fallback : it->second;
    }

    double number(const std::string& key, double fallback) const
    {
        const auto it = values.find(key);
        return it == values.end() ? fallback : std::stod(it->second);
    }

    bool boolean(const std::string& key, bool fallback = false) const
    {
        const auto it = values.find(key);
        if (it == values.end()) {
            return fallback;
        }
        return it->second == "1" || it->second == "true" || it->second == "yes";
    }
};

struct Telemetry
{
    Ring q;
    Ring dq;
    Ring tau;
    Ring tau_ext;
    Ring wrench;
    Ring eef;
    Ring eef_vel;
    Ring status;
    bool world_wrench;

    explicit Telemetry(const Args& args)
    : q(args.get("--q-shm"))
    , dq(args.get("--dq-shm"))
    , tau(args.get("--tau-shm"))
    , tau_ext(args.get("--tau-ext-shm"))
    , wrench(args.get("--wrench-shm"))
    , eef(args.get("--eef-shm"))
    , eef_vel(args.get("--eef-vel-shm"))
    , status(args.get("--status-shm"))
    , world_wrench(args.get("--wrench-frame", "local") == "world")
    {
    }

    void Publish(const flexiv::rdk::RobotStates& states, flexiv::rdk::Robot& robot,
        std::int64_t now_ns, bool publish_status)
    {
        q.Append(states.q, now_ns);
        dq.Append(states.dq, now_ns);
        tau.Append(states.tau, now_ns);
        tau_ext.Append(states.tau_ext, now_ns);
        wrench.Append(world_wrench ? states.ext_wrench_in_world : states.ext_wrench_in_tcp, now_ns);
        eef.Append(states.tcp_pose, now_ns);
        eef_vel.Append(states.tcp_vel, now_ns);
        if (publish_status) {
            const std::array<double, 5> value{
                static_cast<double>(robot.operational_status()),
                robot.estop_released() ? 0.0 : 1.0,
                1.0,
                robot.operational() ? 1.0 : 0.0,
                static_cast<double>(robot.mode()),
            };
            status.Append(value, now_ns);
        }
    }
};

class GripperWorker
{
public:
    GripperWorker(flexiv::rdk::Robot& robot, const Args& args)
    {
        if (!args.has("--gripper-shm") || args.get("--gripper-name", "").empty()) {
            return;
        }
        ring_ = std::make_unique<Ring>(args.get("--gripper-shm"));
        gripper_ = std::make_unique<flexiv::rdk::Gripper>(robot);
        gripper_->Enable(args.get("--gripper-name"));
        if (args.boolean("--gripper-init", true)) {
            gripper_->Init();
        }
        const auto params = gripper_->params();
        open_width_ = args.has("--gripper-open-width")
            ? args.number("--gripper-open-width", params.max_width)
            : params.max_width;
        closed_width_ = args.has("--gripper-closed-width")
            ? args.number("--gripper-closed-width", params.min_width)
            : params.min_width;
        velocity_ = std::clamp(args.number("--gripper-velocity", 0.1), params.min_vel, params.max_vel);
        force_ = std::clamp(args.number("--gripper-force", 20.0), params.min_force, params.max_force);
        deadband_ = args.number("--gripper-deadband", 0.02);
        const auto rate = std::max(0.001, args.number("--gripper-rate", 15.0));
        min_period_ns_ = static_cast<std::int64_t>(1e9 / rate);
        thread_ = std::thread([this] { Run(); });
    }

    GripperWorker(const GripperWorker&) = delete;
    GripperWorker& operator=(const GripperWorker&) = delete;

    ~GripperWorker() { Stop(); }

    void Stop()
    {
        local_stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable()) {
            thread_.join();
        }
        if (gripper_) {
            try {
                gripper_->Stop();
            } catch (...) {
            }
        }
    }

private:
    void Run()
    {
        while (!g_stop.load(std::memory_order_relaxed)
            && !local_stop_.load(std::memory_order_relaxed)) {
            std::vector<double> sample;
            std::int64_t sample_time = 0;
            const auto now = MonotonicNs();
            if (ring_->Latest(sample, sample_time) && !sample.empty()
                && now - last_send_ns_ >= min_period_ns_) {
                const auto value = std::clamp(sample[0], 0.0, 1.0);
                if (!last_value_ || std::abs(value - *last_value_) >= deadband_) {
                    const auto width = open_width_ + value * (closed_width_ - open_width_);
                    try {
                        gripper_->Move(width, velocity_, force_);
                        last_value_ = value;
                        last_send_ns_ = now;
                    } catch (const std::exception& error) {
                        std::cerr << "gripper disabled: " << error.what() << '\n';
                        return;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    std::unique_ptr<Ring> ring_;
    std::unique_ptr<flexiv::rdk::Gripper> gripper_;
    std::thread thread_;
    std::atomic<bool> local_stop_{false};
    double open_width_{0};
    double closed_width_{0};
    double velocity_{0};
    double force_{0};
    double deadband_{0};
    std::int64_t min_period_ns_{0};
    std::int64_t last_send_ns_{0};
    std::optional<double> last_value_;
};

class RobotStopGuard
{
public:
    explicit RobotStopGuard(flexiv::rdk::Robot& robot)
    : robot_(robot)
    {
    }

    ~RobotStopGuard()
    {
        if (armed_) {
            try {
                robot_.Stop();
            } catch (...) {
            }
        }
    }

    void Disarm() { armed_ = false; }

private:
    flexiv::rdk::Robot& robot_;
    bool armed_{true};
};

void TryRealtimeScheduling(int priority)
{
    if (priority <= 0) {
        return;
    }
    sched_param param {};
    param.sched_priority = priority;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        std::cerr << "warning: SCHED_FIFO unavailable; continuing with normal scheduling\n";
    }
}

bool StopCommandPending(Ring& commands, std::int64_t& cursor)
{
    const auto count = commands.write_count();
    if (count - cursor > static_cast<std::int64_t>(commands.capacity())) {
        cursor = count - static_cast<std::int64_t>(commands.capacity());
    }
    std::vector<double> row;
    while (cursor < count) {
        if (!commands.ReadAt(cursor, row)) {
            return false; // torn read: retry this reliable command next tick
        }
        ++cursor;
        if (!row.empty() && static_cast<int>(std::llround(row[0])) == kStopCommand) {
            return true;
        }
    }
    return false;
}

bool ReachedTarget(flexiv::rdk::Robot& robot)
{
    const auto states = robot.primitive_states();
    const auto it = states.find("reachedTarget");
    return it != states.end() && std::holds_alternative<int>(it->second)
        && std::get<int>(it->second) == 1;
}

void AdvanceTrajectory(const std::vector<double>& desired_q, std::vector<double>& target_q,
    std::vector<double>& target_dq, std::vector<double>& target_ddq, double max_joint_vel,
    double max_joint_acc)
{
    constexpr double dt = 0.001;
    for (std::size_t i = 0; i < desired_q.size(); ++i) {
        const auto delta = desired_q[i] - target_q[i];
        const auto previous_velocity = target_dq[i];
        if (std::abs(delta) < 1e-7 && std::abs(previous_velocity) <= max_joint_acc * dt) {
            target_q[i] = desired_q[i];
            target_dq[i] = 0.0;
            target_ddq[i] = -previous_velocity / dt;
            continue;
        }
        const auto braking_speed = std::sqrt(2.0 * max_joint_acc * std::abs(delta));
        const auto speed = std::min(max_joint_vel, braking_speed);
        const auto desired_velocity = std::copysign(speed, delta);
        const auto velocity_delta = std::clamp(desired_velocity - previous_velocity,
            -max_joint_acc * dt, max_joint_acc * dt);
        target_dq[i] = previous_velocity + velocity_delta;
        target_ddq[i] = velocity_delta / dt;
        target_q[i] += target_dq[i] * dt;
    }
}

int ValidateTrajectory()
{
    constexpr double max_velocity = 1.5;
    constexpr double max_acceleration = 2.0;
    std::vector<double> desired(7, 1.0);
    std::vector<double> q(7, 0.0);
    std::vector<double> dq(7, 0.0);
    std::vector<double> ddq(7, 0.0);
    for (int tick = 0; tick < 4000; ++tick) {
        AdvanceTrajectory(desired, q, dq, ddq, max_velocity, max_acceleration);
        for (std::size_t i = 0; i < q.size(); ++i) {
            if (std::abs(dq[i]) > max_velocity + 1e-9
                || std::abs(ddq[i]) > max_acceleration + 1e-9) {
                throw std::runtime_error("trajectory self-test exceeded a motion limit");
            }
        }
    }
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (std::abs(q[i] - desired[i]) > 1e-5 || std::abs(dq[i]) > 1e-5) {
            throw std::runtime_error("trajectory self-test did not converge");
        }
    }
    std::cout << "ok trajectory\n";
    return 0;
}

int ValidateRing(const Args& args)
{
    Ring ring(args.get("--validate-ring"));
    const auto expected = static_cast<std::size_t>(std::stoul(args.get("--expected-dim")));
    if (ring.dim() != expected) {
        throw std::runtime_error("ring dimension does not match --expected-dim");
    }
    std::vector<double> latest;
    std::int64_t timestamp = 0;
    if (!ring.Latest(latest, timestamp)) {
        throw std::runtime_error("ring has no readable sample");
    }
    if (args.boolean("--roundtrip", false)) {
        ring.Append(latest, timestamp + 1);
    }
    std::cout << "ok dim=" << ring.dim() << " count=" << ring.write_count()
              << " timestamp=" << timestamp << '\n';
    return 0;
}

int Run(const Args& args)
{
    Ring setpoints(args.get("--setpoint-shm"));
    Ring commands(args.get("--command-shm"));
    Telemetry telemetry(args);
    const auto dof = static_cast<std::size_t>(std::stoul(args.get("--dof", "7")));
    if (setpoints.dim() < dof) {
        throw std::runtime_error("setpoint ring is narrower than robot DoF");
    }

    flexiv::rdk::Robot robot(args.get("--serial"), {}, args.boolean("--verbose", false));
    RobotStopGuard stop_guard(robot);
    if (robot.info().DoF != dof) {
        throw std::runtime_error("configured DoF does not match connected robot");
    }
    if (robot.fault() && !robot.ClearFault()) {
        throw std::runtime_error("failed to clear robot fault");
    }
    robot.Enable();
    const auto connect_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!robot.operational()) {
        if (g_stop.load(std::memory_order_relaxed)) {
            return 0;
        }
        if (std::chrono::steady_clock::now() > connect_deadline) {
            throw std::runtime_error("robot did not become operational within 30 seconds");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::vector<double> setpoint;
    std::int64_t setpoint_time = 0;
    if (!setpoints.Latest(setpoint, setpoint_time)) {
        throw std::runtime_error("no initial setpoint");
    }
    setpoint.resize(dof);
    auto measured = robot.states();
    if (args.boolean("--safety-check", true)) {
        double worst = 0;
        for (std::size_t i = 0; i < dof; ++i) {
            worst = std::max(worst, std::abs(setpoint[i] - measured.q.at(i)));
        }
        const auto tolerance = args.number("--tolerance", 0.5);
        if (worst > tolerance) {
            throw std::runtime_error("initial joint error exceeds safety tolerance");
        }
    }

    robot.SwitchMode(flexiv::rdk::Mode::NRT_PRIMITIVE_EXECUTION);
    std::array<double, flexiv::rdk::kSerialJointDoF> target_deg {};
    for (std::size_t i = 0; i < std::min(dof, target_deg.size()); ++i) {
        target_deg[i] = setpoint[i] * 180.0 / M_PI;
    }
    robot.ExecutePrimitive("MoveJ", {{"target", flexiv::rdk::JPos(target_deg)}}, true);
    std::int64_t command_cursor = 0;
    while (!ReachedTarget(robot)) {
        const auto now = MonotonicNs();
        telemetry.Publish(robot.states(), robot, now, true);
        if (g_stop.load(std::memory_order_relaxed)
            || StopCommandPending(commands, command_cursor)) {
            robot.Stop();
            stop_guard.Disarm();
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::unique_ptr<GripperWorker> gripper;
    try {
        gripper = std::make_unique<GripperWorker>(robot, args);
    } catch (const std::exception& error) {
        std::cerr << "gripper disabled: " << error.what() << '\n';
    }

    robot.SwitchMode(flexiv::rdk::Mode::RT_JOINT_POSITION);
    std::vector<double> desired_q = setpoint;
    std::vector<double> target_q = setpoint;
    std::vector<double> target_dq(dof, 0.0);
    std::vector<double> target_ddq(dof, 0.0);
    const auto soft_deadman_ns = static_cast<std::int64_t>(args.number("--deadman-ms", 100.0) * 1e6);
    const auto hard_deadman_ns = static_cast<std::int64_t>(args.number("--deadman-hard-ms", 500.0) * 1e6);
    const auto safety_check = args.boolean("--safety-check", true);
    const auto tolerance = args.number("--tolerance", 0.5);
    const auto max_joint_vel = args.number("--max-joint-vel", 2.0);
    const auto max_joint_acc = args.number("--max-joint-acc", 3.0);
    if (max_joint_vel <= 0.0 || max_joint_acc <= 0.0) {
        throw std::invalid_argument("joint velocity/acceleration limits must be positive");
    }
    const auto rt_priority = static_cast<int>(args.number("--rt-priority", 0));
    TryRealtimeScheduling(rt_priority);

    auto next_tick = std::chrono::steady_clock::now();
    std::uint64_t loop_count = 0;
    auto last_setpoint_time = setpoint_time;
    while (!g_stop.load(std::memory_order_relaxed)) {
        next_tick += kLoopPeriod;
        const auto now = MonotonicNs();
        const auto states = robot.states();
        telemetry.Publish(states, robot, now, loop_count % 100 == 0);
        if (robot.fault() || StopCommandPending(commands, command_cursor)) {
            break;
        }

        std::vector<double> newest;
        std::int64_t newest_time = 0;
        if (setpoints.Latest(newest, newest_time)) {
            last_setpoint_time = newest_time;
        }
        const auto age = now - last_setpoint_time;
        if (age > hard_deadman_ns) {
            std::cerr << "hard deadman: setpoint stale for " << age / 1e6 << " ms\n";
            break;
        }
        if (!newest.empty()) {
            if (age <= soft_deadman_ns && newest.size() >= dof) {
                std::copy_n(newest.begin(), dof, desired_q.begin());
            }
        }

        // RDK's RT mode has no internal NRT trajectory generator. Turn each
        // latest-wins position step into a bounded 1 kHz trajectory, including
        // physically meaningful velocity and acceleration feedforward.
        AdvanceTrajectory(desired_q, target_q, target_dq, target_ddq,
            max_joint_vel, max_joint_acc);
        if (safety_check) {
            double worst = 0;
            for (std::size_t i = 0; i < dof; ++i) {
                worst = std::max(worst, std::abs(target_q[i] - states.q.at(i)));
            }
            if (worst > tolerance) {
                std::cerr << "safety halt: joint error " << worst << " rad\n";
                break;
            }
        }
        robot.StreamJointPosition(target_q, target_dq, target_ddq);
        ++loop_count;
        std::this_thread::sleep_until(next_tick);
        if (std::chrono::steady_clock::now() - next_tick > std::chrono::milliseconds(10)) {
            next_tick = std::chrono::steady_clock::now();
        }
    }

    if (gripper) {
        gripper->Stop();
    }
    robot.Stop();
    stop_guard.Disarm();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);
    try {
        const Args args(argc, argv);
        if (args.has("--validate-trajectory")) {
            return ValidateTrajectory();
        }
        if (args.has("--validate-ring")) {
            return ValidateRing(args);
        }
        return Run(args);
    } catch (const std::exception& error) {
        std::cerr << "dfc-flexiv-rt-controller: " << error.what() << '\n';
        return 1;
    }
}
