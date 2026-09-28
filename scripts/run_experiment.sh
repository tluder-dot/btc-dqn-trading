#!/usr/bin/env bash
# Train one experiment config with several seeds in parallel (one CPU thread each).
# Usage: scripts/run_experiment.sh configs/<exp>.json [seed ...]   (default seeds: 0 1 2)
set -euo pipefail
config="$1"; shift
if [ $# -gt 0 ]; then seeds=("$@"); else seeds=(0 1 2); fi
exp_id=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['exp_id'])" "$config")
mkdir -p "runs/$exp_id"
for seed in "${seeds[@]}"; do
  python -m rl_trading.train --config "$config" --seed "$seed" > "runs/$exp_id/seed_$seed.log" 2>&1 &
done
wait
echo "finished $exp_id (seeds: ${seeds[*]})"
