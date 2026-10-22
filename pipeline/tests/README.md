# Tests

**`test_*.py` run on the host** under `.venvs/episode0`, with no container:

```bash
.venvs/episode0/bin/python -m pytest pipeline/tests/ -q
```

**`container_*.py` need Warp, JAX and NVIDIA's `flash_chord`**, which exist only
inside `flash_chord:latest`. They are deliberately *not* named `test_*` so a
plain host run collects nothing that cannot import.

A module-level `pytest.importorskip` was tried first and is not sufficient:
`pipeline/tests` is a package, so a skip raised while importing one module
aborts collection of the whole package and the host suite silently reports
"collected 0 items". A `conftest.py` with `collect_ignore` does not work either
— it is loaded, but pytest does not apply it to files inside a package
directory. Naming is the only mechanism that reliably works here.

pytest collects explicitly-named files regardless of the `python_files`
pattern, so the container runs them by path:

```bash
docker run --rm --gpus all \
  -v $PWD/official/video_to_data/robotic_grounding/flash_chord:/workspace \
  -v $PWD/pipeline:/v2d/repo/pipeline:ro \
  -v $PWD/outputs/step_D/warp_cache:/root/.cache/warp \
  -e PYTHONPATH=/v2d/repo -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -w /v2d/repo flash_chord:latest \
  python -m pytest \
    pipeline/tests/container_flash_sac_orientation_kernel.py \
    pipeline/tests/container_flash_sac_orientation_replay.py -q -p no:cacheprovider
```

`bash scripts/verify_orientation_flash_sac.sh` runs both suites.
