from __future__ import annotations

import sys

from weather_ml import time_feature_sweep


def _default_experiment_ids() -> list[str]:
    return [f"E{num:02d}" for num in range(51, 101)]


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--experiment-ids" not in argv:
        argv += ["--experiment-ids"] + _default_experiment_ids()
    if "--sweep-id" not in argv:
        argv += ["--sweep-id", "sweep_004"]
    raise SystemExit(time_feature_sweep.main(argv))
