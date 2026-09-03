#!/bin/bash
# Drive the rest of the 12-task ID90 pipeline: datagen -> train -> eval scenes -> eval.
#
# Two failures of the previous driver (mg_autosubmit_train.sh) are fixed here.
#
#  1. It ran as a login-node background process, so it died with the shell that started it.
#     mug_cleanup_d1's datagen finished afterwards and nothing picked it up. This one is meant
#     to run inside a SLURM job (mg_pipeline_supervisor.job) and outlives any session.
#
#  2. Every sbatch was fire-and-forget. The gpu QOS caps a user at MaxSubmitJobsPU=100 queued
#     jobs; while the training arrays sat pending that cap was reached, so the per-task
#     `sbatch mg_gen_eval_scenes_id90.job` calls were REJECTED and merely logged -- which is why
#     9 of 12 tasks had no eval scenes. Here every submission checks for room first, verifies
#     the returned job id, and is retried on the next pass if it did not go through.
#
# All state is on disk (stats json, scene hdf5s, epoch-500 checkpoints, marker files), so this
# can be killed and resubmitted at any point without redoing work.
set -u
cd /scratch1/hyeonhoo/code/Robomimic_Async
PY=/scratch1/hyeonhoo/miniconda3/envs/robot_mimic_mg/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

TASKS=(stack_d1 stack_three_d1 square_d2 threading_d0 three_piece_assembly_d0 hammer_cleanup_d1 \
       mug_cleanup_d1 coffee_d2 kitchen_d1 pick_place_d0 coffee_preparation_d1 nut_assembly_d0)
ARMS=(baseline aux_world_frame aux_eef_frame aux_obj_eef_frame)
RUNGS=(id ood_pos ood_yaw ood_both)
RES=/scratch1/hyeonhoo/results
STATE=$RES/id90_train/.autosubmit
mkdir -p "$STATE"

CAP=${CAP:-100}                 # gpu QOS MaxSubmitJobsPU
ACCTS=(gaurav_1048 biyik_1173)  # biyik_1165 is the worst of the three; see carc-slurm-accounts
POLL=${POLL:-300}
N_EVAL=12                       # eval array elements per task: 4 arms x 3 seeds

queued () { squeue -u "$USER" -h 2>/dev/null | wc -l; }

# submit <n_elements> <sbatch args...> -- refuses unless the array fits under the cap with margin
submit () {
    local need=$1; shift
    local free=$(( CAP - $(queued) ))
    if [ "$free" -lt $((need + 3)) ]; then
        echo "    deferred: needs $need slots, $free free under the $CAP cap"; return 1
    fi
    local out; out=$(sbatch --parsable "$@" 2>&1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then echo "$out"; return 0; fi
    echo "    sbatch REJECTED: $out"; return 1
}

acct ()   { echo "${ACCTS[$(( $1 % ${#ACCTS[@]} ))]}"; }
horizon () { $PY -c "import json;print(json.load(open('robomimic/exps/templates/id90/${1}_id90_diffusion_84.json'))['experiment']['rollout']['horizon'])" 2>/dev/null || echo 400; }

trained () {    # all four arms of a task carry an epoch-500 checkpoint
    local t=$1 a
    for a in "${ARMS[@]}"; do
        ls "$RES/id90_train/$t/${t}_${a}_id90_diffusion_84"/*/models/model_epoch_500*.pth >/dev/null 2>&1 || return 1
    done
}

scened () {     # all four scene sets exist
    local t=$1 r
    for r in "${RUNGS[@]}"; do [ -f "$RES/id90_eval_scenes/$t/scenes_${r}.hdf5" ] || return 1; done
}

echo "=== supervisor up on $(hostname) at $(date) | cap $CAP, poll ${POLL}s ==="

while :; do
    outstanding=0
    for i in "${!TASKS[@]}"; do
        T=${TASKS[$i]}; A=$(acct "$i")

        # --- stage 1: training, once datagen has written its final stats ------------------
        if [ ! -f "$STATE/$T.submitted" ] && [ ! -f "$STATE/$T.failed" ]; then
            if [ ! -f "$RES/mg_${T}_id90/${T}_id90/important_stats.json" ]; then
                outstanding=$((outstanding + 1)); continue          # datagen still running
            fi
            echo "[$(date +%H:%M:%S)] $T: datagen done -> config from real episode lengths"
            $PY mg_make_train_configs.py --smoke_ok 2>&1 | grep -E "^$T " || true
            if ! $PY mg_check_obj_blocks.py "$T"; then
                echo "[$(date +%H:%M:%S)] $T: BLOCK CHECK FAILED -- not submitting"
                touch "$STATE/$T.failed"; continue
            fi
            HZ=$(horizon "$T"); TLIM=03:00:00; [ "$HZ" -gt 800 ] && TLIM=04:00:00
            if JID=$(submit 4 --array=$((i*4))-$((i*4+3)) --time=$TLIM --account=$A mg_train_id90_12task.job); then
                echo "[$(date +%H:%M:%S)] $T: SUBMITTED train $JID (horizon $HZ, $TLIM, $A)"
                touch "$STATE/$T.submitted"
            fi
            outstanding=$((outstanding + 1)); continue
        fi
        [ -f "$STATE/$T.failed" ] && continue

        # --- stage 2: eval scenes; independent of training, env_meta comes from the data ---
        if ! scened "$T"; then
            if [ ! -f "$STATE/$T.scenes" ]; then
                if SJID=$(submit 1 --array=$i --account=$A mg_gen_eval_scenes_id90.job); then
                    echo "[$(date +%H:%M:%S)] $T: SUBMITTED scenes $SJID"; touch "$STATE/$T.scenes"
                fi
            elif ! squeue -u "$USER" -h -n mg_scenes -o "%K" 2>/dev/null | grep -qx "$i"; then
                echo "[$(date +%H:%M:%S)] $T: scene job gone but scenes incomplete -- will retry"
                rm -f "$STATE/$T.scenes"
            fi
            outstanding=$((outstanding + 1)); continue
        fi

        # --- stage 3: evaluation, once every arm has reached epoch 500 ---------------------
        [ -f "$STATE/$T.eval" ] && continue
        if ! trained "$T"; then outstanding=$((outstanding + 1)); continue; fi
        HZ=$(horizon "$T"); ETL=06:00:00; [ "$HZ" -gt 800 ] && ETL=12:00:00
        if EJID=$(submit $N_EVAL --array=0-$((N_EVAL-1)) --time=$ETL --account=$A \
                         --export=ALL,TASK=$T mg_eval_id90.job); then
            echo "[$(date +%H:%M:%S)] $T: SUBMITTED eval $EJID ($N_EVAL x [4 rungs x 10 ckpts], $ETL, $A)"
            touch "$STATE/$T.eval"
        fi
        outstanding=$((outstanding + 1))
    done

    if [ "$outstanding" -eq 0 ]; then echo "[$(date +%H:%M:%S)] every task has its eval submitted"; break; fi
    echo "[$(date +%H:%M:%S)] $outstanding task(s) outstanding | $(queued) jobs queued of $CAP"
    sleep "$POLL"
done

echo "=== SUMMARY $(date) ==="
for T in "${TASKS[@]}"; do
    s=""
    [ -f "$STATE/$T.submitted" ] && s="train"
    [ -f "$STATE/$T.failed" ]    && s="CHECK FAILED"
    scened "$T"                  && s="$s+scenes"
    [ -f "$STATE/$T.eval" ]      && s="$s+eval"
    printf "%-24s %s\n" "$T" "${s:-waiting on datagen}"
done
