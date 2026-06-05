# Setup

This page records the recommended environment for VideoSEG-O3.

## 1. Create Environment

```bash
conda create -n videoseg-o3 python=3.10 -y
conda activate videoseg-o3
```

## 2. Install Requirements

```bash
pip install -U pip
pip install -r requirement.txt
```

The pinned requirements are taken with Python 3.10 and CUDA 12.4. If you use a different CUDA version, install the matching PyTorch build first and then install the remaining packages.

```bash
# Optional: install FlashAttention after PyTorch is available.
pip install flash-attn==2.7.1.post4 --no-build-isolation
```

## 3. Prepare Pretrained Models

The current VideoSEG-O3 configs use Qwen3-VL as the base MLLM and SAM2 as the
mask decoder. Place the pretrained weights under `pretrained/`:

```text
pretrained/
├── Qwen3-VL-2B-Instruct/
├── Qwen3-VL-4B-Instruct/
└── sam2_hiera_large.pt
```

Download the base MLLM checkpoints from the official Qwen releases:

- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [SAM2-Hiera-Large](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

One possible download pattern is:

```bash
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir pretrained/Qwen3-VL-2B-Instruct

huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir pretrained/Qwen3-VL-4B-Instruct

curl -L \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
  -o pretrained/sam2_hiera_large.pt
```

## 4. Prepare Data

Follow [Data Preparation](DATA.md) to organize training and evaluation datasets
under `data/`.
