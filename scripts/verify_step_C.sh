#!/usr/bin/env bash
# Verify Step C - Step A + Step B generalized to every episode of one tier.
# Run from the repo root:
#     bash scripts/verify_step_C.sh                 # tier 1, checks only (~90 s)
#     bash scripts/verify_step_C.sh --tier tier_2_synthetic_noise
#     bash scripts/verify_step_C.sh --rerun         # also re-runs the whole pipeline (~2 h)
# Every check prints PASS or FAIL. Exit code 0 means everything passed.
set -uo pipefail
cd "$(dirname "$0")/.."

TIER=tier_1_multiview_caption
RERUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --rerun) RERUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PY=.venvs/episode0/bin/python
OUT="outputs/$TIER"
REPORT="$OUT/report.json"
INDEX="$OUT/references_index.json"
EXPECTED_EPISODES=30

FAILED=0
ok(){ echo "  PASS  $1"; }
no(){ echo "  FAIL  $1"; FAILED=1; }
chk(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else no "$1"; fi; }
jq_py(){ "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print($2)" "$1" 2>/dev/null; }

echo "=== 0. tier under test: $TIER ==="

if [ "$RERUN" -eq 1 ]; then
  echo
  echo "=== re-running the whole tier pipeline (stage -> report) ==="
  "$PY" scripts/run_tier_pipeline.py all --tier "$TIER" || no "run_tier_pipeline all exited non-zero"
  echo
  echo "=== re-running the Episode 0 regression ==="
  "$PY" scripts/run_tier_pipeline.py regress --tier "$TIER" || no "the Episode 0 regression failed"
fi

echo
echo "=== 1. unit tests for the registry and the adapter ==="
if "$PY" -m pytest pipeline/tests -q > /tmp/step_c_pytest.log 2>&1; then
  ok "pytest pipeline/tests: $(tail -1 /tmp/step_c_pytest.log)"
else
  no "pytest pipeline/tests failed (see /tmp/step_c_pytest.log)"
fi

echo
echo "=== 2. the tier mirror is complete and the registry agrees with it ==="
chk "registry loads $TIER" \
    "$PY -c \"import sys;sys.path.insert(0,'.');from pipeline.episodes import load_tier;load_tier('$TIER')\""
n_eps=$("$PY" -c "import sys;sys.path.insert(0,'.');from pipeline.episodes import load_tier;print(len(load_tier('$TIER')))" 2>/dev/null)
[ "${n_eps:-0}" = "$EXPECTED_EPISODES" ] \
  && ok "$TIER declares $EXPECTED_EPISODES episodes and every input file resolves" \
  || no "expected $EXPECTED_EPISODES episodes, registry reports ${n_eps:-none}"

echo
echo "=== 3. reports exist ==="
chk "report.json present"            "[ -f '$REPORT' ]"
chk "report.md present"              "[ -f '$OUT/report.md' ]"
chk "references_index.json present"  "[ -f '$INDEX' ]"
if [ ! -f "$REPORT" ]; then
  echo
  echo "  cannot continue without $REPORT - run:"
  echo "    $PY scripts/run_tier_pipeline.py all --tier $TIER"
  exit 1
fi

echo
echo "=== 4. every episode passed every stage ==="
total=$(jq_py "$REPORT" "d['episodes_total']")
passing=$(jq_py "$REPORT" "d['episodes_passing']")
failing=$(jq_py "$REPORT" "d['failing_indices']")
[ "$total" = "$EXPECTED_EPISODES" ] \
  && ok "the report covers all $EXPECTED_EPISODES episodes" \
  || no "the report covers $total episodes, expected $EXPECTED_EPISODES"
[ "$passing" = "$total" ] \
  && ok "$passing/$total episodes passed stage + retarget + validate + flashchord" \
  || no "$passing/$total episodes passed; failing: $failing"

for stage in stage retarget validate flashchord; do
  bad=$(jq_py "$REPORT" "[e['episode_index'] for e in d['episodes'] if not (e.get('$stage') or {}).get('ok')]")
  [ "$bad" = "[]" ] && ok "$stage: no failures" || no "$stage failed for episodes $bad"
done

echo
echo "=== 5. frame counts and timestamps were preserved ==="
src=$(jq_py "$REPORT" "d['source_frames_total']")
ref=$(jq_py "$REPORT" "d['reference_frames_total']")
dropped=$(jq_py "$REPORT" "sum(len(e['dropped_frames']) for e in d['episodes'])")
echo "  info: $src source frames -> $ref reference frames ($dropped deliberately excluded)"
"$PY" - "$REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
bad = []
for entry in report["episodes"]:
    expected = entry["source_frames"] - len(entry["dropped_frames"])
    if expected != entry["reference_frames"]:
        bad.append((entry["episode_index"], expected, entry["reference_frames"]))
    got = ((entry.get("validate") or {}).get("metrics") or {}).get("frames")
    if got is not None and got != entry["reference_frames"]:
        bad.append((entry["episode_index"], entry["reference_frames"], got))
print("  PASS  every reference has exactly source_frames - dropped_frames rows"
      if not bad else f"  FAIL  frame-count mismatches: {bad}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || FAILED=1

echo
echo "=== 6. the support decision is backed by two independent counts ==="
"$PY" - "$REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
bad, enabled = [], 0
for entry in report["episodes"]:
    fc = entry.get("flashchord") or {}
    if not fc:
        bad.append((entry["episode_index"], "no flashchord record"))
        continue
    if fc.get("usd_collision_shapes") != entry["support_shape_count"]:
        bad.append((entry["episode_index"], "USD vs host scan disagree"))
    if fc.get("include_support") != entry["scene_support"]:
        bad.append((entry["episode_index"], "scene.support disagrees"))
    enabled += bool(entry["scene_support"])
print(f"  info: scene.support=true for {enabled} of {len(report['episodes'])} episodes")
print("  PASS  the host text scan and FlashCHORD's USD parse agree on every episode"
      if not bad else f"  FAIL  {bad}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || FAILED=1

echo
echo "=== 7. Episode 0 regression against the preserved baseline ==="
base="$OUT/regression/episode_000000_baseline.parquet"
chk "preserved baseline parquet exists" "[ -f '$base' ]"
reg_ok=$(jq_py "$REPORT" "(d.get('regression') or {}).get('ok')")
reg_delta=$(jq_py "$REPORT" "(d.get('regression') or {}).get('worst_abs_delta')")
if [ "$reg_ok" = "True" ]; then
  ok "Episode 0 re-run matches the baseline (worst delta $reg_delta)"
else
  no "the Episode 0 regression has not passed (run: $PY scripts/run_tier_pipeline.py regress --tier $TIER)"
fi

echo
echo "=== 8. tier isolation ==="
"$PY" - "$TIER" <<'PY'
import sys
sys.path.insert(0, ".")
from pipeline.episodes import load_tier, TIER_1, TIER_2
tier = sys.argv[1]
other = TIER_2 if tier == TIER_1 else TIER_1
mine = load_tier(tier)
theirs = {s.sequence_id: s for s in load_tier(other)}
clashes = [
    s.sequence_id for s in mine
    if s.sequence_id in theirs
    and (s.data_folder == theirs[s.sequence_id].data_folder
         or s.reference_parquet == theirs[s.sequence_id].reference_parquet)
]
print(f"  info: {tier} and {other} share {len(set(theirs) & {s.sequence_id for s in mine})} sequence ids")
print("  PASS  no staged path is shared between the two tiers"
      if not clashes else f"  FAIL  shared staged paths: {clashes}")
sys.exit(1 if clashes else 0)
PY
[ $? -eq 0 ] || FAILED=1
# The parquet bakes an absolute mesh path; it must name this tier's tree.
"$PY" - "$INDEX" <<'PY'
import json, sys
index = json.load(open(sys.argv[1]))
roots = {e["container_root"] for e in index["episodes"]}
print(f"  info: container mount point(s) for this tier: {sorted(roots)}")
print("  PASS  this tier mounts at exactly one tier-specific container path"
      if len(roots) == 1 else f"  FAIL  mixed mount points {roots}")
sys.exit(0 if len(roots) == 1 else 1)
PY
[ $? -eq 0 ] || FAILED=1

echo
echo "=== 9. upstream untouched ==="
if [ -z "$(git -C official/video_to_data status --short)" ]; then
  ok "NVIDIA upstream unmodified"; else no "NVIDIA upstream has local changes"; fi
chk "upstream still pinned at 193382a" \
    "git -C official/video_to_data log --oneline -1 | grep -q 193382a"

echo
echo "=== 10. runtime and memory, as measured ==="
"$PY" - "$REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
seconds = report["seconds"]
print("  info: retarget {:.0f} s total, {:.0f} s median per episode".format(
    seconds["retarget"],
    sorted(float((e.get("retarget") or {}).get("seconds") or 0)
           for e in report["episodes"])[len(report["episodes"]) // 2]))
print("  info: flashchord {:.0f} s total".format(seconds["flashchord"]))
print("  info: peak GPU delta  retarget {} MiB / flashchord {} MiB".format(
    report["retarget_peak_gpu_delta_mib"], report["flashchord_peak_gpu_delta_mib"]))
print("  info: peak container RAM during retarget {:.0f} MiB".format(
    report["retarget_peak_container_mib"]))
PY
echo -n "  info: disk free: "; df -h / | awk 'NR==2{print $4" ("$5" used)"}'
echo -n "  info: GPU now:   "; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo
if [ "$FAILED" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit $FAILED
