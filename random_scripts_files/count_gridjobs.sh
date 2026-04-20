BASE_DIR="/home/raubenb/gitrepos/QTF/grid_runs/grid_v3_rot_pi_largegrid"

TOTAL=3888

DONE=$(find "$BASE_DIR" -type f -path "*/beam/beamsearch_best.json" | wc -l)

REMAINING=$((TOTAL - DONE))

echo "Total jobs:     $TOTAL"
echo "Completed jobs: $DONE"
echo "Remaining jobs: $REMAINING"
