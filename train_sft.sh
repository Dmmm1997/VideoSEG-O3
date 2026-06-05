#!/bin/bash

CONFIG_LIST=(
  "projects/sa2va/configs/videoseg_o3/videoseg_o3_2b_sft.py"
  # "projects/sa2va/configs/videoseg_o3/videoseg_o3_4b_sft.py"
  # "projects/sa2va/configs/videoseg_o3/videoseg_o3_2b_coldstart.py"
  # "projects/sa2va/configs/videoseg_o3/videoseg_o3_4b_coldstart.py"
)

for config in "${CONFIG_LIST[@]}"; do
      config_name=$(basename "$config" .py)
      echo ">>> Processing config: $config_name"

      # ---------------------------------------------------------
      # 1. 启动训练
      # ---------------------------------------------------------
      bash tools/dist.sh train "$config" 8

      # 获取训练生成的 checkpoint 路径
      pth_path=$(cat work_dirs/"$config_name"/last_checkpoint)
      
      # 定义转换后的输出路径
      hf_output_path=work_dirs/$config_name/hf_model

      # ---------------------------------------------------------
      # 2. 转换为 HF 格式
      # ---------------------------------------------------------
      python tools/convert_to_hf.py \
        "$config" \
        --pth-model $pth_path \
        --save-path $hf_output_path

      echo "---------------------------------------------"
done