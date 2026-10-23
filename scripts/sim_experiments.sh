#!/usr/bin/env bash
# Closed-loop evaluation and MAPPO fine-tuning in the Arena-1 simulator.
#
#     ARENA1_ROOT=/path/to/Arena-1 bash scripts/sim_experiments.sh
#
# Two sources of policies:
#   * REAL (default when present): the runs trained on the official dataset by
#     scripts/run_all.sh — rm_runs/infantry_{bc,iql,dt}_tactical + data/vis_map.npz.
#   * SIM=1: stand-in runs trained by the very same pipeline on matches the
#     Arena-1 example strategies play against each other (exported to the
#     official SQLite schema).  Use it to exercise everything without the DJI data.
#
# Stages (each skipped when its output exists, so a re-run resumes):
#   0. [SIM] export scripted matches -> vis_map -> team_prior -> dataset -> BC/IQL/DT
#   1. parity check: live 161-D observation == offline build_obs
#   2. closed-loop evaluation of script / random / BC / IQL / DT
#   3. MAPPO fine-tuning from BC (shared actor, centralised critic, KL to BC)
#   4. closed-loop evaluation of the fine-tuned actor on the same grid
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ARENA1_ROOT:?set ARENA1_ROOT to an Arena-1 checkout with a built engine}"
export ARENA1_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1     # many processes share the CPU; avoid thread thrash
WORKERS="${WORKERS:-$(python -c 'import os;print(os.cpu_count() or 1)')}"
SIM="${SIM:-0}"
GAMES="${GAMES:-110}"                # scripted matches for the SIM dataset
SEEDS="${SEEDS:-2}"                  # evaluation seeds per (opponent, side)
OPPONENTS="${OPPONENTS:-claude_opus55_xhigh_hold,claude_opus55_xhigh_team,siege_team}"
MAPPO_ITERS="${MAPPO_ITERS:-30}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [[ "$SIM" == "1" ]]; then
  P=arena_; RUNS=rm_runs/arena
  DB=data/arena_sim.sqlite; VIS=data/arena_vis_map.npz; TP=data/arena_team_prior.json
  DATA=data/arena_infantry_tactical
  log "exporting $GAMES scripted Arena-1 matches to $DB"
  python -m rm_rl.sim.export_sqlite --out "$DB" --games "$GAMES" --workers "$WORKERS"
  [[ -f "$VIS" ]] || python -m rm_rl.data.vis_map --db "$DB" --out "$VIS"
  [[ -f "$TP" ]] || python -m rm_rl.data.team_prior --db "$DB" --out "$TP"
  [[ -f "$DATA/meta.json" ]] || python -m rm_rl.data.build_dataset --db "$DB" --out "$DATA" \
      --agent infantry --config configs/infantry_iql_tactical.yaml --action-mode tactical \
      --goal-horizon 5 --vis-map "$VIS" --team-prior "$TP"
  python -m rm_rl.data.check_dataset --data "$DATA" || true
  for a in bc iql; do
    [[ -f "$RUNS/$a/best.pt" ]] || python -m rm_rl.train.train_offline \
        --config "configs/infantry_${a}_tactical.yaml" --data "$DATA" --out "$RUNS/$a"
  done
  [[ -f "$RUNS/dt/best.pt" ]] || python -m rm_rl.train.train_dt \
      --config configs/infantry_dt_tactical.yaml --data "$DATA" --out "$RUNS/dt"
  BC=$RUNS/bc; IQL=$RUNS/iql; DT=$RUNS/dt
else
  RUNS=rm_runs; VIS=data/vis_map.npz
  BC=rm_runs/infantry_bc_tactical; IQL=rm_runs/infantry_iql_tactical; DT=rm_runs/infantry_dt_tactical
  for d in "$BC" "$IQL" "$DT"; do
    [[ -f "$d/best.pt" ]] || { echo "missing $d/best.pt (run scripts/run_all.sh, or SIM=1)" >&2; exit 1; }
  done
  [[ -f "$VIS" ]] || { echo "missing $VIS (the engagement map the runs were trained with)" >&2; exit 1; }
fi

log "parity: live observation vs offline build_obs"
python -m rm_rl.sim.check_parity --duration 60 --vis-map "$VIS"

CL=$RUNS/closed_loop
log "closed-loop evaluation -> $CL"
python -m rm_rl.eval.closed_loop --out "$CL" --vis-map "$VIS" --workers "$WORKERS" \
  --opponents "$OPPONENTS" --seeds "$SEEDS" \
  --arm script --arm random --arm "bc=$BC" --arm "iql=$IQL" --arm "dt=$DT" --arm "bc_vision=$BC:vision"

MAPPO=$RUNS/mappo_from_bc
log "MAPPO fine-tuning from BC -> $MAPPO"
python -m rm_rl.train.train_mappo --init "$BC" --vis-map "$VIS" --out "$MAPPO" \
  --iters "$MAPPO_ITERS" --workers "$WORKERS" --resume

log "closed-loop evaluation of the fine-tuned actor"
python -m rm_rl.eval.closed_loop --out "$CL" --vis-map "$VIS" --workers "$WORKERS" \
  --opponents "$OPPONENTS" --seeds "$SEEDS" --arm "mappo=$MAPPO/latest"

log "done: $CL/summary.md"
cat "$CL/summary.md"
