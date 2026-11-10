#!/usr/bin/env bash
# Start a permutation-test run inside an existing tmux window, one pane per worker,
# plus a pane that refreshes the p-value report every 15 minutes.
#
#   scripts/permutation_tmux.sh <tmux-window> <workers> <hours> [out-folder]
#   scripts/permutation_tmux.sh 0:3 10 18
#
# All workers share one deadline. Stopping a pane (Ctrl-C) is safe: rerun this
# script with the same out-folder and every worker resumes where it left off.
set -euo pipefail

window=${1:?tmux window, e.g. 0:3}
workers=${2:?number of workers}
hours=${3:?hours to run}
out=${4:-results/permutation}

project=$(cd "$(dirname "$0")/.." && pwd)
deadline=$(python3 -c "import time; print(int(time.time() + $hours * 3600))")
env="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"

first_pane=$(tmux list-panes -t "$window" -F '#{pane_id}' | head -1)
run_in() {  # pane-id command
    tmux send-keys -t "$1" "cd '$project' && $2" Enter
}

run_in "$first_pane" "$env uv run permutation-test --worker 0 --until $deadline --out $out"
for ((w = 1; w < workers; w++)); do
    pane=$(tmux split-window -t "$window" -c "$project" -P -F '#{pane_id}')
    tmux select-layout -t "$window" tiled >/dev/null
    run_in "$pane" "$env uv run permutation-test --worker $w --until $deadline --out $out"
done

monitor=$(tmux split-window -t "$window" -c "$project" -P -F '#{pane_id}')
tmux select-layout -t "$window" tiled >/dev/null
sleep_s=900
run_in "$monitor" "sleep 120; while true; do clear; date; uv run permutation-report --out $out 2>&1 | head -40; [ \$(date +%s) -ge $deadline ] && break; sleep $sleep_s; done"

echo "Started $workers workers in $window until $(date -d @"$deadline")"
echo "Results: $project/$out  (report any time: uv run permutation-report --out $out)"
