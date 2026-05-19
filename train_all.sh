#!/usr/bin/env bash
set -euo pipefail

# Optional: change to the directory where these scripts live
# cd /path/to/your/project

SCRIPTS=(
  "train_dl3p.sh"
  "train_segf.sh"
  "train_segnx.sh"
  "train_unet3p.sh"
)

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# One timestamp for the whole batch so it's easy to group
BATCH_TS="$(date +%Y%m%d_%H%M%S)"

for s in "${SCRIPTS[@]}"; do
  if [[ ! -x "$s" ]]; then
    echo "Making $s executable"
    chmod +x "$s"
  fi

  base="$(basename "$s" .sh)"
  log="$LOG_DIR/${base}_${BATCH_TS}.log"

  echo "[$(date --iso-8601=seconds)] Starting $s -> $log"
  # Run sequentially; tee shows progress if you're attached, still logs everything.
  bash "$s" 2>&1 | tee -a "$log"
  echo "[$(date --iso-8601=seconds)] Finished  $s"
done

echo "All trainings completed. Batch: $BATCH_TS"

# Running in background:
# chmod +x train_all.sh
# nohup bash ./train_all.sh > logs/batch_nohup_$(date +%Y%m%d_%H%M%S).out 2>&1 &

# Monitoring logs:
# tail -f logs/batch_nohup_*.out
# ls -lt logs/