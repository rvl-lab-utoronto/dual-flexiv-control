#!/usr/bin/env python3
"""Scan a U2D2/DYNAMIXEL bus for servos across common baud rates.

Usage:
    python scripts/dxl_scan.py [PORT]

PORT defaults to /dev/ttyUSB0. Requires read/write access to the port
(dialout group membership, or run under sudo).

Protocol 2.0 uses a fast broadcast ping (returns all responders at once).
Protocol 1.0 has no broadcast ping, so IDs 0..252 are pinged individually.
"""
import sys

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# X-series/2.0 default is 57600; older gear and custom setups vary. Ordered
# most-likely-first so a hit prints early.
BAUDS = [57600, 1000000, 2000000, 3000000, 115200, 9600, 4000000, 4500000]


def scan_proto2(port_name, baud):
    port = PortHandler(port_name)
    ph = PacketHandler(2.0)
    if not port.openPort():
        raise RuntimeError(f"could not open {port_name}")
    try:
        port.setBaudRate(baud)
        ids, _ = ph.broadcastPing(port)  # {id: (model_no, fw_ver)}
        found = []
        for dxl_id in ids:
            model, comm, err = ph.ping(port, dxl_id)
            found.append((dxl_id, model))
        return found
    finally:
        port.closePort()


def scan_proto1(port_name, baud, id_range=range(0, 253)):
    port = PortHandler(port_name)
    ph = PacketHandler(1.0)
    if not port.openPort():
        raise RuntimeError(f"could not open {port_name}")
    try:
        port.setBaudRate(baud)
        found = []
        for dxl_id in id_range:
            model, comm, err = ph.ping(port, dxl_id)
            if comm == COMM_SUCCESS:
                found.append((dxl_id, model))
        return found
    finally:
        port.closePort()


def main():
    port_name = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    print(f"Scanning {port_name}\n")
    any_found = False
    for baud in BAUDS:
        # Protocol 2.0 (fast broadcast).
        try:
            hits = scan_proto2(port_name, baud)
        except Exception as e:
            print(f"[{baud:>7} 2.0] ERROR: {e}")
            hits = []
        for dxl_id, model in hits:
            any_found = True
            print(f"[{baud:>7} 2.0] id={dxl_id:<3} model={model}")

        # Protocol 1.0 (per-ID; only bother if 2.0 found nothing at this baud).
        if not hits:
            try:
                hits1 = scan_proto1(port_name, baud)
            except Exception as e:
                print(f"[{baud:>7} 1.0] ERROR: {e}")
                hits1 = []
            for dxl_id, model in hits1:
                any_found = True
                print(f"[{baud:>7} 1.0] id={dxl_id:<3} model={model}")

    print("\n" + ("Done." if any_found else "No servos responded on any baud rate."))


if __name__ == "__main__":
    main()
