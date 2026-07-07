"""ssh_hosts: the eval launcher's Host-alias discovery over ~/.ssh/config."""

from dual_flexiv_control.dashboard.ssh_hosts import SshHost
from dual_flexiv_control.dashboard.ssh_hosts import discover_ssh_hosts


def test_missing_config_yields_no_hosts(tmp_path):
    assert discover_ssh_hosts(tmp_path / "nope") == []


def test_hosts_resolve_to_hostname_and_wildcards_are_skipped(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text(
        "# GPU boxes\n"
        "Host borabora\n"
        "    HostName 100.92.86.90\n"
        "    User rvl_root\n"
        "\n"
        "Host jeju github.com-work\n"
        "  HostName 100.85.217.39\n"
        "\n"
        "Host bare-alias\n"
        "    User someone\n"
        "\n"
        "Host *\n"
        "    ServerAliveInterval 60\n"
    )
    got = discover_ssh_hosts(cfg)
    assert got == [
        SshHost("borabora", "100.92.86.90"),
        SshHost("jeju", "100.85.217.39"),
        SshHost("github.com-work", "100.85.217.39"),
        # No HostName: ssh dials the alias itself.
        SshHost("bare-alias", "bare-alias"),
    ]


def test_first_block_wins_and_key_eq_value_form_parses(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text(
        "Host dup\n"
        "    HostName=first.example\n"
        "Host dup\n"
        "    HostName second.example\n"
    )
    assert discover_ssh_hosts(cfg) == [SshHost("dup", "first.example")]


def test_include_glob_is_followed(tmp_path):
    (tmp_path / "config.d").mkdir()
    (tmp_path / "config.d" / "gpu").write_text(
        "Host molokini\n    HostName 10.0.0.7\n"
    )
    cfg = tmp_path / "config"
    cfg.write_text(
        "Include config.d/*\n"
        "Host local\n    HostName 127.0.0.1\n"
    )
    assert discover_ssh_hosts(cfg) == [
        SshHost("molokini", "10.0.0.7"),
        SshHost("local", "127.0.0.1"),
    ]


def test_match_block_closes_open_host(tmp_path):
    cfg = tmp_path / "config"
    cfg.write_text(
        "Host open-ended\n"
        "Match user root\n"
        "    HostName should.not.attach\n"
    )
    assert discover_ssh_hosts(cfg) == [SshHost("open-ended", "open-ended")]
