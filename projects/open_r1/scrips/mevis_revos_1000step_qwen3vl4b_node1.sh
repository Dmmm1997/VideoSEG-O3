export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_LAUNCH_BLOCKING=1

#!/bin/bash
# =================================================================
NNODES=2
NODE_RANK=0
NPROC_PER_NODE=8
# =================================================================

MASTER_ADDR=localhost
MASTER_PORT=12345

# MODIFY HERE: please prepare the env related variables
PR1_PATH="projects/open_r1"
CHECKPOINT_PATH="./work_dirs_RL" # directory to save the checkpoint
RUN_NAME="final_version/qwen3vl4b_1000step" # exp name
MODEL_PATH="work_dirs/sa2va_qwen3_4b_coldstart/hf_model"

# Default Setting
OUTPUT_DIR="${CHECKPOINT_PATH}/${RUN_NAME}" # path to save the output
SRC_PATH="${OUTPUT_DIR}/src" # path to backup the source code

export LOG_DIR="${OUTPUT_DIR}/logs" # path to save the log

export WANDB_DIR="${OUTPUT_DIR}"
export WANDB_PROJECT="COT_SEG" # project name in wandb
export WANDB_TAGS="qwen3_img_rl" # tags for the experiment in wandb
export WANDB_MODE=offline 

if [ ! -d "${OUTPUT_DIR}"/src ]; then
    mkdir -p ${OUTPUT_DIR}/src
fi

if [ ! -d "${WANDB_DIR}" ]; then
    mkdir -p ${WANDB_DIR}
fi

# backup the source code
cp -r ${PR1_PATH} ${SRC_PATH}
mkdir -p ${LOG_DIR}

# run the training
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${PR1_PATH}/grpo_vllm_sa2va_r1_cot.py \
    --deepspeed projects/open_r1/configs/zero2.json \
    --dataset_name "mevis,revos" \
    --dataset_type default \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path ${MODEL_PATH} \
    --max_prompt_length 8192 \
    --max_completion_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 16 \
    --num_generations 4 \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --max_pixels 200704 \
    --max_steps 1000 \
    --run_name ${RUN_NAME} \
    --save_strategy "steps" \
    --save_steps 1000 \
    --reward_funcs "strict_format_cot" "step_format_cot" "lr_vtg_iou_dense" "lr_vtg_keyframe" "lr_maskiou_dense" "lr_keyframe_vs_mask_dense" "lr_progressive_keyframe_dense" \
    --save_only_model true \
    --system_prompt_template "seg_cot" \
    --question_template "seg_temporal_cot" \
    --train_sample_size 100000 \
    --skip_special_tokens false \
    --answer_template "default" \
    --learning_rate 3e-6 \
    --if_detach_res_loss false \
    --if_use_visualization false \
    --use_mask_logps true \
    --res_loss_ratio 0.2 \
    --vis_interval 10 \
    --beta 0.04 \
    --max_turn 3

