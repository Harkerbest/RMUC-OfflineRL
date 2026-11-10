#!/usr/bin/env bash
# 2027 sentry decision layer in Arena-1: self-test -> baselines -> training -> evaluation.
#
#     ARENA1_ROOT=/path/to/Arena-1 RULES=arena bash scripts/sentry27.sh          # everything
#     STAGE=train ARENA1_ROOT=... RULES=arena bash scripts/sentry27.sh           # one stage
#
# RULES decides what the policy may learn (docs/sentry27/README.md, "偷基地"):
#   arena         Arena-1's rules: a robot on the enemy base card takes the enemy base
#                 position (+3 control/s).  The policy starts from the raid tree and is
#                 measured against it.
#   conservative  the real rules do not allow that capture: the enemy_base goal, the raid
#                 tree and the base-raid team scripts are all removed; the policy starts
#                 from the balanced tree and is measured against it.
#
# Stages (each resumes where it stopped):
#   selftest  pre-flight checks (engine, features, masks, a short match)
#   baseline  paired evaluation of the reference arms on held-out team scripts
#   train     tree imitation, then PPO self-play with an opponent league
#   eval      the trained policy (greedy) and checkpoints against the trees
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ARENA1_ROOT:?set ARENA1_ROOT to an Arena-1 checkout with a built engine}"
: "${RULES:?set RULES=arena or RULES=conservative (see the header)}"
export ARENA1_ROOT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CPUS=$(python -c 'import os; print(os.cpu_count() or 1)')
TRAIN_WORKERS="${TRAIN_WORKERS:-$((CPUS > 16 ? CPUS - 8 : CPUS - 1))}"
EVAL_WORKERS="${EVAL_WORKERS:-$((CPUS > 8 ? CPUS - 4 : CPUS))}"
ITERS="${ITERS:-300}"
SEEDS="${SEEDS:-2}"
OUT="${OUT:-rm_runs/sentry27_v2_$RULES}"        # v1 (the first 300-update run) lives in rm_runs/sentry27
EVAL="${EVAL:-rm_runs/sentry27_v2_${RULES}_eval}"
STAGE="${STAGE:-all}"
CKPT_EVERY="${CKPT_EVERY:-50}"

case "$RULES" in
  arena)        TEACHER=raid;     REF=bt:raid;     RULE_FLAGS=();
                TREES=(--arm bt:balanced --arm bt:guard --arm bt:assault --arm bt:raid) ;;
  conservative) TEACHER=balanced; REF=bt:balanced; RULE_FLAGS=(--no-base-raids);
                TREES=(--arm bt:balanced --arm bt:guard --arm bt:assault) ;;
  *) echo "RULES must be arena or conservative" >&2; exit 1 ;;
esac

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
want() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }
evaluate() {
  python -m rm_rl.eval.eval_sentry27 --out "$EVAL" --workers "$EVAL_WORKERS" --seeds "$SEEDS" --ref "$REF" \
    "${RULE_FLAGS[@]}" --arm script "${TREES[@]}" --arm random "$@"
}

if want selftest; then
  log "self-test"
  python -m rm_rl.arena27.selftest
fi

if want baseline; then
  log "baselines -> $EVAL (RULES=$RULES, paired, held-out team scripts, $SEEDS seeds x 2 sides)"
  evaluate
fi

if want train; then
  log "training -> $OUT (RULES=$RULES, teacher bt:$TEACHER, $TRAIN_WORKERS workers, $ITERS updates)"
  python -m rm_rl.train.train_sentry27 --out "$OUT" --workers "$TRAIN_WORKERS" --iters "$ITERS" \
    --imitate "$TEACHER" "${RULE_FLAGS[@]}" --export-every 10 --resume
fi

if want eval; then
  arms=(--arm "rl=$OUT/latest")
  for d in "$OUT"/iter_*; do
    n=$((10#${d##*_}))
    (( n % CKPT_EVERY == 0 )) && arms+=(--arm "rl_$(printf %04d "$n")=$d")
  done
  log "evaluation of the trained policy -> $EVAL"
  evaluate "${arms[@]}"
  python -m rm_rl.arena27.selftest --run "$OUT/latest"
  cat "$EVAL/summary.md"
fi
