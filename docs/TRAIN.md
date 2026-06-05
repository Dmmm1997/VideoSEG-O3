# Training

VideoSEG-O3 uses three stages: SFT, CoT cold start, and RL. Run all commands
from the repository root.

## Stage 1/2: SFT and CoT Cold Start

Use the root training launcher:

```bash
bash train_sft.sh
```

`train_sft.sh` reads configs from `CONFIG_LIST`. Edit this list to select the
stage and model scale you want to train:

```text
projects/sa2va/configs/videoseg_o3/videoseg_o3_2b_sft.py
projects/sa2va/configs/videoseg_o3/videoseg_o3_4b_sft.py
projects/sa2va/configs/videoseg_o3/videoseg_o3_2b_coldstart.py
projects/sa2va/configs/videoseg_o3/videoseg_o3_4b_coldstart.py
```

For CoT cold start, edit the `pretrained_pth` field in the selected config so
it points to the converted SFT checkpoint.

After each config finishes, `train_sft.sh` reads:

```text
work_dirs/<config_name>/last_checkpoint
```

and converts the checkpoint to Hugging Face format with:

```text
tools/convert_to_hf.py
```

The converted model is written to:

```text
work_dirs/<config_name>/hf_model
```

## Stage 3: RL

Use the root RL launcher:

```bash
bash train_rl.sh
```

Before launching, update these variables in `train_rl.sh`:

```bash
RUN_NAME="videoseg-o3/qwen3vl2b_1000step"
MODEL_PATH="/path/to/coldstart/hf_model"
NPROC_PER_NODE=8
```

`MODEL_PATH` should point to a released or locally trained CoT Hugging Face
checkpoint. The script writes checkpoints and logs under:

```text
work_dirs_RL/<RUN_NAME>/
```

For multi-node RL, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`
consistently in `train_rl.sh` on each machine.

The RL launcher calls:

```text
projects/open_r1/grpo_vllm_sa2va_r1_cot.py
```


## Outputs

```text
work_dirs/       # SFT and CoT checkpoints
work_dirs_RL/    # RL checkpoints
```
