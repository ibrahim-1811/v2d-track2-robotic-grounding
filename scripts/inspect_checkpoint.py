"""Print a FlashCHORD checkpoint's identity using NVIDIA's own metadata reader.

Run inside the flash_chord container with the checkpoint reachable:

    python inspect_checkpoint.py <path/to/model_N.safetensors>

Emits KEY=value lines so shell checks can grep them without parsing JSON.
"""

from __future__ import annotations

import sys

from flash_chord.evaluation.harness import (
    checkpoint_config,
    checkpoint_training_algorithm,
)
from flash_chord.training.checkpoint import read_checkpoint_metadata


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: inspect_checkpoint.py <checkpoint.safetensors>", file=sys.stderr)
        return 2
    metadata = read_checkpoint_metadata(sys.argv[1])
    config = checkpoint_config(metadata)

    print("ALGO {}".format(checkpoint_training_algorithm(config)))
    print("ITER {}".format(metadata.get("iteration")))
    print("STEPS {}".format(metadata.get("environment_steps")))
    print("EMBODIMENT {}".format(config["embodiment"]["_target_"]))
    print("TRAINING_TARGET {}".format(config["training"]["_target_"]))
    print("WORLD_COUNT {}".format(config["scene"]["world_count"]))
    print("MAX_ITERATIONS {}".format(config["training"]["max_iterations"]))
    print("PARQUET {}".format(config["task"]["parquet"]))
    print("TRAINED_VOC_SCALE {}".format(float(config["env"]["voc_scale"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
