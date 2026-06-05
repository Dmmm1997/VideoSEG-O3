# Data Preparation

This page describes the expected dataset layout for VideoSEG-O3. Run training
and evaluation commands from the repository root so that relative paths resolve
against `./data`.

## Data Sources

Download the released annotation files from
[dmmm997/VideoSEG-O3](https://modelscope.cn/datasets/dmmm997/VideoSEG-O3). This
release contains the VTS-CoT cold-start annotations and the JSON annotations
used for RL training.

For the original video segmentation datasets, masks, and evaluation assets,
please refer to the companion data preparation repository:
[dmmm997/MomentSeg](https://modelscope.cn/datasets/dmmm997/MomentSeg), or the
official sources of each benchmark. Place or symlink the downloaded files under
`data/` following the layout below.

## Directory Layout

After downloading, organize the data as:

```text
data/
├── video_datas/
│   ├── revos/
│   ├── mevis/
│   ├── davis17/
│   ├── chat_univi/
│   ├── sam_v_full/
│   ├── sam_v_final_custom.json
│   ├── Long-RVOS/
│   ├── ref_sav_eval/
│   ├── GroundMoRe/
│   └── reasonvos/
├── ref_seg/
│   ├── refclef/
│   ├── refcoco/
│   ├── refcoco+/
│   └── refcocog/
├── ref_sav/
│   └── Ref-SAV.json
├── reason_seg/
├── glamm_data/
│   ├── images/
│   └── annotations/
├── osprey-724k/
│   ├── Osprey-724K/
│   └── coco/
├── llava_data/
│   ├── llava_images/
│   ├── LLaVA-Instruct-150K/
│   └── LLaVA-Pretrain/
├── VTG/
│   ├── NumPro_FT/
│   └── TimeLens-100K/
├── VTG-CoT/
│   ├── longrvos.json
│   ├── mevis.json
│   └── revos.json
└── VideoSEG-O3-RL/
    ├── rl_mevis_data.json
    └── rl_revos_data.json
```

## Sa2VA Training Data

VideoSEG-O3 is trained with a three-stage data recipe.

| Stage | Capability | Training data |
| --- | --- | --- |
| Stage I: SFT | Video QA, image/video segmentation, temporal index understanding | ChatUniVi, TimeLens-100K, RefCOCO/+/g, ReasonSeg, GCG, Ref-YTVOS, ReVOS, MeViS, Ref-SAV, Long-RVOS |
| Stage II: CoT Cold-Start | Step-wise reasoning and tool usage | VTS-CoT, curated from ReVOS, Long-RVOS, and MeViS |
| Stage III: GRPO-RL | Multi-turn interaction and mask refinement | ReVOS and MeViS |

For Sa2VA-format SFT data, follow the upstream Sa2VA data preparation
instruction and place the extracted files under `data/`. The original video
segmentation datasets can be prepared by following the companion dataset
repository referenced above.

```bash
mkdir -p data
# Download the required original datasets and place or symlink them under data/.
# Keep the folder names consistent with the layout above.
```

The SA-V video dataset is not included in Sa2VA training archives. Download it
from the official Segment Anything Video source and place it at:

```text
data/video_datas/sam_v_full/
```

## VTS-CoT Annotations

Download the released annotation files and place the VTS-CoT files under:

```text
data/VTG-CoT/
├── longrvos.json
├── mevis.json
└── revos.json
```

## RL Training Data

Download the released annotation files and place the RL JSON files under:

```text
data/VideoSEG-O3-RL/
├── rl_mevis_data.json
└── rl_revos_data.json
```

The current RL entry point reads:

```text
data/VideoSEG-O3-RL/rl_mevis_data.json
data/VideoSEG-O3-RL/rl_revos_data.json
```

If you use a different location, update the dataset paths in
`projects/open_r1/grpo_vllm_sa2va_r1_cot.py` before launching training.

## Evaluation Data

Evaluation scripts cover ReVOS, MeViS, DAVIS, Ref-SAV, ReasonVOS, Long-RVOS,
and GroundMoRe. Place the original datasets and benchmark assets in the
matching folders under `data/video_datas/`:

```text
data/video_datas/revos/
data/video_datas/mevis/
data/video_datas/davis17/
data/video_datas/ref_sav_eval/
data/video_datas/reasonvos/
data/video_datas/Long-RVOS/
data/video_datas/GroundMoRe/
```

For submission-style datasets, the evaluation command will write predictions to
the selected `--work_dir`, usually under:

```text
<MODEL_DIR>/evaluation/<DATASET>/
```
