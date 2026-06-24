#!/usr/bin/env python
"""Launch the bimanual Flexiv brain (Hydra entry point).

Configuration is composed from ``src/dual_flexiv_control/conf`` and validated
against the structured schema in ``dual_flexiv_control.configs``. Override anything
on the CLI, Hydra-style:

    # Hardware-free smoke run (10 s, simulated sources)
    python scripts/run_system.py runtime.sim=true runtime.duration_s=10

    # Live run against two arms
    python scripts/run_system.py \
        arms.left.serial=Rizon4-XXXXXX arms.right.serial=Rizon4-YYYYYY

    # Pick a controller per arm / retune rates
    python scripts/run_system.py control@arms.left.control=force brain.rate_hz=200

    # See the fully composed config without running
    python scripts/run_system.py --cfg job

This is a thin wrapper over ``dual_flexiv_control.system.main`` (also installed as
the ``dual-flexiv-control`` console script).
"""

from dual_flexiv_control.system import main

if __name__ == "__main__":
    main()
