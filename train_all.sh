#!/bin/bash
set -euo pipefail

scripts=(
  "train_dl3p.sh"
  "train_segf.sh"
  "train_segnx.sh"
  "train_unet3p.sh"
)

for s in "${scripts[@]}"; do
  echo "=============================="
  echo "Starting: $s  ($(date))"
  echo "=============================="

  bash "$s"

  echo "Finished: $s  ($(date))"
  echo
done

echo "All trainings completed ($(date))"