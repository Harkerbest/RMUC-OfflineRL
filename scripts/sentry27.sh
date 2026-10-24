#!/usr/bin/env bash
# 2027 sentry decision layer in Arena-1: self-test -> baselines -> training -> evaluation.
#
#     ARENA1_ROOT=/path/to/Arena-1 bash scripts/sentry27.sh            # everything
#     STAGE=train ARENA1_ROOT=... bash scripts/sentry27.sh              # one stage
#
# Stages (each resumes where it stopped; see docs/sentry27/README.md):
#   selftest  pre-flight checks (engine, features, masks, a short match)
#   baseline  paired evaluation of the reference arms: team script, the three
#             behaviour trees, random — on held-out team scripts
#   train     tree imitation, then PPO self-play with an opponent league
#   eval      the trained policy (greedy) and checkpoints against the trees
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ARENA1_ROOT:?set ARENA1_ROOT to an Arena-1 checkout with a built engine}"
export ARENA1_ROOT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CPUS=$(python -c 'import os; print(os.cpu_count() or 1)')
TRAIN_WORKERS="${TRAIN_WORKERS:-$((CPUS > 16 ? CPUS - 8 : CPUS - 1))}"
EVAL_WORKERS="${EVAL_WORKERS:-$((CPUS > 8 ? CPUS - 4 : CPUS))}"
ITERS="${ITERS:-300}"
SEEDS="${SEEDS:-2}"
OUT="${OUT:-rm_runs/sentry27}"
EVAL="${EVAL:-rm_runs/sentry27_eval}"
STAGE="${STAGE:-all}"
CKPT_EVERY="${CKPT_EVERY:-50}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
want() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }

if want selftest; then
  log "self-test"
  python -m rm_rl.arena27.selftest
fi

if want baseline; then
  log "baselines -> $EVAL (paired, held-out team scripts, $SEEDS seeds x 2 sides)"
  python -m rm_rl.eval.eval_sentry27 --out "$EVAL" --workers "$EVAL_WORKERS" --seeds "$SEEDS" \
    --arm script --arm bt:balanced --arm bt:guard --arm bt:assault --arm random
fi

if want train; then
  log "training -> $OUT ($TRAIN_WORKERS workers, $ITERS updates)"
  python -m rm_rl.train.train_sentry27 --out "$OUT" --workers "$TRAIN_WORKERS" --iters "$ITERS" \
    --export-every 10 --resume
fi

if want eval; then
  arms=(--arm "rl=$OUT/latest")
  for d in "$OUT"/iter_*; do
    n=$((10#${d##*_}))
    (( n % CKPT_EVERY == 0 )) && arms+=(--arm "rl_$(printf %04d "$n")=$d")
  done
  log "evaluation of the trained policy -> $EVAL"
  python -m rm_rl.eval.eval_sentry27 --out "$EVAL" --workers "$EVAL_WORKERS" --seeds "$SEEDS" \
    --arm script --arm bt:balanced --arm bt:guard --arm bt:assault --arm random "${arms[@]}"
  python -m rm_rl.arena27.selftest --run "$OUT/latest"
  cat "$EVAL/summary.md"
fi
