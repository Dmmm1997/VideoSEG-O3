#!/bin/bash

set -e

MODEL_DIRS=(
    work_dirs_RL/final_version/videoseg-o3-2b-rl/checkpoint-1000
)

NUM_GPUS=${NUM_GPUS:-4}

declare -A EVAL_SCRIPT_MAP
EVAL_SCRIPT_MAP["REVOS"]="tools/eval/eval_revos.py"
EVAL_SCRIPT_MAP["MEVIS_U"]="tools/eval/eval_mevis.py"
EVAL_SCRIPT_MAP["DAVIS"]="tools/eval/eval_davis.py"
EVAL_SCRIPT_MAP["REF_SAV"]="tools/eval/eval_ref_sav.py"
EVAL_SCRIPT_MAP["REASONVOS"]="tools/eval/eval_reasonvos.py"
EVAL_SCRIPT_MAP["LONGRVOS"]="tools/eval/eval_longrvos.py"
EVAL_SCRIPT_MAP["GROUNDMORE"]="tools/eval/eval_groundmore.py"

DATASETS=(
    "MEVIS_U"
    "DAVIS"
    "REASONVOS"
    "REF_SAV"
    "REVOS"
    "REFYTVOS"
    "MEVIS"
    "LONGRVOS"
    "GROUNDMORE"
)

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        PY_SCRIPT=${EVAL_SCRIPT_MAP[$DATASET]}
        WORK_DIR="${MODEL_DIR}/evaluation/${DATASET}"

        echo "------------------------------------------------"
        echo "Model:   ${MODEL_DIR}"
        echo "Dataset: ${DATASET}"

        if [ -n "$PY_SCRIPT" ]; then
            echo "Mode: metric evaluation (${PY_SCRIPT})"

            bash projects/sa2va/evaluation/dist_test.sh \
                projects/sa2va/evaluation/sa2va_eval_ref_vos_cot_v2.py \
                "$MODEL_DIR" "$NUM_GPUS" \
                --work_dir "$WORK_DIR" \
                --dataset "$DATASET" \
                --max_turns 3 \
                --max_video_sample 20 \
                --max_temporal_frames_per_round 5 \
                --max_select_K 8

            RESULTS_FILE="${WORK_DIR}/results.json"
            python "$PY_SCRIPT" "$RESULTS_FILE" --save_name "lang_inj_${DATASET}.json"
        else
            echo "Mode: submission generation"

            bash projects/sa2va/evaluation/dist_test.sh \
                projects/sa2va/evaluation/sa2va_eval_ref_vos_cot_v2.py \
                "$MODEL_DIR" "$NUM_GPUS" \
                --work_dir "$WORK_DIR" \
                --dataset "$DATASET" \
                --max_turns 3 \
                --max_video_sample 20 \
                --max_temporal_frames_per_round 5 \
                --max_select_K 8 \
                --submit

            if [ -d "${WORK_DIR}/Annotations" ]; then
                echo "Zipping submission results in ${WORK_DIR}"
                pushd "$WORK_DIR" > /dev/null
                zip -qr "${DATASET}.zip" Annotations/
                popd > /dev/null
            else
                echo "Warning: ${WORK_DIR}/Annotations not found; skipping zip."
            fi
        fi

        echo "Finished: ${MODEL_DIR} - ${DATASET}"
    done
done
