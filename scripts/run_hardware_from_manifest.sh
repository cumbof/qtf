#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MANIFEST.csv [ROW_INDEX|all]" >&2
  exit 2
fi

MANIFEST=$1
ROW_INDEX=${2:-${SLURM_ARRAY_TASK_ID:-1}}
CALLER_CWD=$PWD
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
if [[ "$MANIFEST" != /* ]]; then
  MANIFEST="$CALLER_CWD/$MANIFEST"
fi
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
QTF_BIN=${QTF_BIN:-qtf}
SHOTS=${QTF_HW_SHOTS:-8192}
OPT_LEVEL=${QTF_HW_OPT_LEVEL:-3}

if [[ "$ROW_INDEX" == "all" ]]; then
  ROW_COUNT=$(
    "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(newline="") as handle:
    print(sum(1 for _ in csv.DictReader(handle)))
PY
  )
  for row in $(seq 1 "$ROW_COUNT"); do
    "$0" "$MANIFEST" "$row"
  done
  exit 0
fi

read -r PARAMS_JSON OUT_PDB OUT_JSON REFERENCE_PDB JOB_ID < <(
  "$PYTHON_BIN" - "$MANIFEST" "$ROW_INDEX" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
row_index = int(sys.argv[2])
with manifest.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if row_index < 1 or row_index > len(rows):
    raise SystemExit(f"ROW_INDEX must be 1..{len(rows)}; got {row_index}")
row = rows[row_index - 1]
required = ("params_json", "out_pdb", "out_json", "job_id")
missing = [name for name in required if not row.get(name)]
if missing:
    raise SystemExit(f"Manifest row is missing required values: {', '.join(missing)}")
print(
    row["params_json"],
    row["out_pdb"],
    row["out_json"],
    row.get("reference_pdb") or "-",
    row["job_id"],
)
PY
)

if [[ "${QTF_HW_SKIP_EXISTING:-0}" == "1" && -s "$OUT_JSON" ]]; then
  echo "Skipping existing hardware job ${ROW_INDEX}: ${JOB_ID}"
  echo "Found output JSON: ${OUT_JSON}"
  exit 0
fi

RUN_ROOT=$(dirname "$(dirname "$OUT_JSON")")
GROMACS_DIR="$RUN_ROOT/gromacs_minimized/$JOB_ID"
cmd=(
  "$QTF_BIN" fold-hardware
  --params-json "$PARAMS_JSON"
  --out-pdb "$OUT_PDB"
  --out-json "$OUT_JSON"
  --gromacs-outdir "$GROMACS_DIR"
  --rmsd_mode ca
  --rmsd_residue_scope core
  --shots "$SHOTS"
  --optimization-level "$OPT_LEVEL"
)

if [[ "$REFERENCE_PDB" != "-" ]]; then
  cmd+=(--reference_pdb "$REFERENCE_PDB")
fi
if [[ "${QTF_HW_USE_AER:-0}" == "1" ]]; then
  cmd+=(--local-simulator)
fi
if [[ -n "${QTF_HW_BACKEND_NAME:-}" ]]; then
  cmd+=(--backend-name "$QTF_HW_BACKEND_NAME")
fi
if [[ -n "${QTF_HW_CHANNEL:-}" ]]; then
  cmd+=(--channel "$QTF_HW_CHANNEL")
fi
if [[ -n "${QTF_HW_INSTANCE:-}" ]]; then
  cmd+=(--instance "$QTF_HW_INSTANCE")
fi
if [[ -n "${QTF_HW_TOKEN:-}" ]]; then
  cmd+=(--token "$QTF_HW_TOKEN")
fi
if [[ -n "${QTF_HW_SEED_TRANSPILE:-}" ]]; then
  cmd+=(--seed-transpiler "$QTF_HW_SEED_TRANSPILE")
fi
if [[ "${QTF_HW_SAMPLER_MAX_MITIGATION:-1}" == "0" ]]; then
  cmd+=(--no-sampler-max-mitigation)
fi
if [[ "${QTF_HW_GROMACS:-1}" == "0" ]]; then
  cmd+=(--no-gromacs)
fi

echo "Running hardware job ${ROW_INDEX}: ${JOB_ID}"
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"

