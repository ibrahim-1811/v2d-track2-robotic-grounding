"""Prove the official evaluator strips object assistance from a checkpoint's config.

Applies NVIDIA's own ``apply_evaluation_composition`` to the config embedded in a
checkpoint and prints the Virtual Object Control settings before and after. The
replay path runs with ``voc_scale=1.0`` (an external PD wrench plus gravity
compensation on the object); evaluation must force it to 0.0, so an assisted
replay can never be mistaken for policy success.

    python check_unassisted.py <path/to/model_N.safetensors>
"""

from __future__ import annotations

import sys

from flash_chord.evaluation.harness import (
    apply_evaluation_composition,
    checkpoint_config,
)
from flash_chord.training.checkpoint import read_checkpoint_metadata


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_unassisted.py <checkpoint.safetensors>", file=sys.stderr)
        return 2
    config = checkpoint_config(read_checkpoint_metadata(sys.argv[1]))

    trained_scale = float(config["env"]["voc_scale"])
    print("TRAINED {} {} {}".format(
        trained_scale,
        float(config["env"]["voc"]["max_force"]),
        float(config["env"]["voc"]["max_torque"]),
    ))

    apply_evaluation_composition(config, world_count=1, reset_mode="explicit")
    eval_scale = float(config["env"]["voc_scale"])
    eval_force = float(config["env"]["voc"]["max_force"])
    eval_torque = float(config["env"]["voc"]["max_torque"])
    print("EVAL {} {} {}".format(eval_scale, eval_force, eval_torque))
    print("AUTO_RESET {}".format(config["env"]["auto_reset"]))
    print("WORLD_COUNT {}".format(config["scene"]["world_count"]))

    unassisted = eval_scale == 0.0 and eval_force == 0.0 and eval_torque == 0.0
    print("UNASSISTED {}".format("YES" if unassisted else "NO"))
    return 0 if unassisted else 1


if __name__ == "__main__":
    sys.exit(main())
