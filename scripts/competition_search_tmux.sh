#!/usr/bin/env bash
# Run the competition settings search on shuffled histories, one tmux pane per worker,
# plus a pane that refreshes the report every 10 minutes.
#
#   scripts/competition_search_tmux.sh <tmux-window> <workers> <hours> [out-folder]
#   scripts/competition_search_tmux.sh 0:3 9 3
#
# Uses the panes already in the window (extra panes are created as needed).
set -euo pipefail

window=${1:?tmux window, e.g. 0:3}
workers=${2:?number of workers}
hours=${3:?hours to run}
out=${4:-results/competition_search}

project=$(cd "$(dirname "$0")/.." && pwd)
deadline=$(python3 -c "import time; print(int(time.time() + $hours * 3600))")
env="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore"

mapfile -t panes < <(tmux list-panes -t "$window" -F '#{pane_id}')
while ((${#panes[@]} < workers + 1)); do
    panes+=("$(tmux split-window -t "$window" -c "$project" -P -F '#{pane_id}')")
    tmux select-layout -t "$window" tiled >/dev/null
done
run_in() { tmux send-keys -t "$1" C-c; tmux send-keys -t "$1" "clear; cd '$project' && $2" Enter; }

for ((w = 0; w < workers; w++)); do
    run_in "${panes[$w]}" "$env uv run competition-search --worker $w --until $deadline --out $out"
done
run_in "${panes[$workers]}" "sleep 90; while true; do clear; date; uv run competition-search-report --out $out --top 20 2>&1 | cut -c1-170; [ \$(date +%s) -ge $deadline ] && break; sleep 600; done"

echo "Started $workers search workers in $window until $(date -d @"$deadline")"
echo "Results: $project/$out  (report any time: uv run competition-search-report --out $out)"
