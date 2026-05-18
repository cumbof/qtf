#!/usr/bin/env bash
set -euo pipefail

PANEL_CSV="${1:-protein_panel.csv}"
OUTROOT="${2:-panel_runs}"
DATE_TAG="${3:-$(date +%Y%m%d_%H%M%S)}"

# Shared settings for this experiment
FORCEFIELD_DEFAULT="amber"
CHI_MODE_DEFAULT="selective"
CHAIN_DEFAULT="A"
BEAM_WIDTH=200
WINDOW_DEG=30
STEP_DEG=15
MAX_SIDECHAIN_OPTS=9
RANDOM_SEED=42

# Current tuning settings
HBOND_SCALE=0.75
SASA_SCALE=0.7
VDW_REP_SCALE=0.1
VDW_ATTR_SCALE=0.1

mkdir -p "${OUTROOT}"

trim() {
  echo "$1" | xargs
}

safe_name() {
  local s="$1"
  s="${s// /_}"
  s="${s//\//_}"
  s="${s//:/_}"
  echo "$s"
}

tail -n +2 "${PANEL_CSV}" | while IFS=, read -r NAME SEQUENCE PDB_PATH CHAIN FORCEFIELD CHI_MODE
do
  NAME="$(trim "${NAME}")"
  SEQUENCE="$(trim "${SEQUENCE}")"
  PDB_PATH="$(trim "${PDB_PATH}")"
  CHAIN="$(trim "${CHAIN}")"
  FORCEFIELD="$(trim "${FORCEFIELD}")"
  CHI_MODE="$(trim "${CHI_MODE}")"

  FORCEFIELD="${FORCEFIELD:-$FORCEFIELD_DEFAULT}"
  CHI_MODE="${CHI_MODE:-$CHI_MODE_DEFAULT}"
  CHAIN="${CHAIN:-$CHAIN_DEFAULT}"

  SAFE_NAME="$(safe_name "${NAME}")"
  ABS_PDB_PATH="$(realpath "${PDB_PATH}")"

  RUN_ID="${SAFE_NAME}_ff-${FORCEFIELD}_chi-${CHI_MODE}_hb-${HBOND_SCALE}_sasa-${SASA_SCALE}_vdwr-${VDW_REP_SCALE}_vdwa-${VDW_ATTR_SCALE}"
  RUN_DIR="${OUTROOT}/${DATE_TAG}/${RUN_ID}"

  mkdir -p "${RUN_DIR}/beam"
  mkdir -p "${RUN_DIR}/native"

  echo "===================================================="
  echo "Running ${NAME}"
  echo "Sequence: ${SEQUENCE}"
  echo "PDB: ${ABS_PDB_PATH}"
  echo "Output: ${RUN_DIR}"
  echo "===================================================="

  if [[ ! -f "${PDB_PATH}" ]]; then
    echo "ERROR: Missing PDB file ${PDB_PATH} for ${NAME}" >&2
    exit 1
  fi

  cat > "${RUN_DIR}/run_settings.txt" <<EOF2
name=${NAME}
sequence=${SEQUENCE}
pdb_path=${ABS_PDB_PATH}
chain=${CHAIN}
forcefield=${FORCEFIELD}
chi_mode=${CHI_MODE}
beam_width=${BEAM_WIDTH}
window_deg=${WINDOW_DEG}
step_deg=${STEP_DEG}
max_sidechain_opts_per_residue=${MAX_SIDECHAIN_OPTS}
random_seed=${RANDOM_SEED}
hbond_scale=${HBOND_SCALE}
sasa_scale=${SASA_SCALE}
vdw_rep_scale=${VDW_REP_SCALE}
vdw_attr_scale=${VDW_ATTR_SCALE}
EOF2

  echo "Beam search..."
  QTF_HBOND_SCALE="${HBOND_SCALE}" \
  QTF_SASA_SCALE="${SASA_SCALE}" \
  QTF_VDW_REP_SCALE="${VDW_REP_SCALE}" \
  QTF_VDW_ATTR_SCALE="${VDW_ATTR_SCALE}" \
  python qtf_beamsearch_benchmark.py \
    --protein_name "${NAME}" \
    --sequence "${SEQUENCE}" \
    --forcefield "${FORCEFIELD}" \
    --beam_width "${BEAM_WIDTH}" \
    --window_deg "${WINDOW_DEG}" \
    --step_deg "${STEP_DEG}" \
    --chi_mode "${CHI_MODE}" \
    --max_sidechain_opts_per_residue "${MAX_SIDECHAIN_OPTS}" \
    --random_seed "${RANDOM_SEED}" \
    --reference_pdb "${ABS_PDB_PATH}" \
    --outdir "${RUN_DIR}/beam"

  echo "Native scoring..."
  QTF_HBOND_SCALE="${HBOND_SCALE}" \
  QTF_SASA_SCALE="${SASA_SCALE}" \
  QTF_VDW_REP_SCALE="${VDW_REP_SCALE}" \
  QTF_VDW_ATTR_SCALE="${VDW_ATTR_SCALE}" \
  python qtf_score_experimental.py \
    --name "${NAME}" \
    --pdb_path "${ABS_PDB_PATH}" \
    --chain "${CHAIN}" \
    --forcefield "${FORCEFIELD}" \
    --chi_mode "${CHI_MODE}" \
    --out_csv "${RUN_DIR}/native/${SAFE_NAME}_native_score.csv" \
    --out_json "${RUN_DIR}/native/${SAFE_NAME}_native_score.json"

  echo "Wrote beam outputs to: ${RUN_DIR}/beam"
  ls -1 "${RUN_DIR}/beam" || true
  echo "Wrote native outputs to: ${RUN_DIR}/native"
  ls -1 "${RUN_DIR}/native" || true
done
