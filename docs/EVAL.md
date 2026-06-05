# Evaluation

Run evaluation commands from the repository root.

## Main Evaluation Launcher

`evaluation.sh` is the main evaluation entry point. Update `MODEL_DIRS` in the
script to point to the checkpoint you want to evaluate, then run:

```bash
bash evaluation.sh
```

By default, the script evaluates each model with:

```text
projects/sa2va/evaluation/dist_test.sh
projects/sa2va/evaluation/sa2va_eval_ref_vos_cot_v2.py
```

Set `NUM_GPUS` if you want to override the default GPU count:

```bash
NUM_GPUS=8 bash evaluation.sh
```

## Manual Evaluation

```bash
bash projects/sa2va/evaluation/dist_test.sh \
  projects/sa2va/evaluation/sa2va_eval_ref_vos_cot_v2.py \
  <MODEL_DIR> <NUM_GPUS> \
  --work_dir <OUTPUT_DIR> \
  --dataset <DATASET_NAME> \
  --max_turns 3 \
  --max_video_sample 20 \
  --max_temporal_frames_per_round 5 \
  --max_select_K 8
```

Example:

```bash
MODEL_DIR=work_dirs_RL/final_version/videoseg-o3-2b/checkpoint-1000
DATASET=REVOS

bash projects/sa2va/evaluation/dist_test.sh \
  projects/sa2va/evaluation/sa2va_eval_ref_vos_cot_v2.py \
  "$MODEL_DIR" 4 \
  --work_dir "$MODEL_DIR/evaluation/$DATASET" \
  --dataset "$DATASET" \
  --max_turns 3 \
  --max_video_sample 20 \
  --max_temporal_frames_per_round 5 \
  --max_select_K 8
```

## Metric Scripts

`evaluation.sh` maps datasets to the standard metric scripts:

```text
tools/eval/eval_davis.py
tools/eval/eval_groundmore.py
tools/eval/eval_longrvos.py
tools/eval/eval_mevis.py
tools/eval/eval_reasonvos.py
tools/eval/eval_ref_sav.py
tools/eval/eval_revos.py
tools/eval/eval_tvg.py
```

Current mapping:

```text
REVOS      -> tools/eval/eval_revos.py
MEVIS_U    -> tools/eval/eval_mevis.py
DAVIS      -> tools/eval/eval_davis.py
REF_SAV    -> tools/eval/eval_ref_sav.py
REASONVOS  -> tools/eval/eval_reasonvos.py
LONGRVOS   -> tools/eval/eval_longrvos.py
GROUNDMORE -> tools/eval/eval_groundmore.py
```

After inference, metric scripts are called on:

```text
<MODEL_DIR>/evaluation/<DATASET>/results.json
```

Example:

```bash
python tools/eval/eval_revos.py \
  "$MODEL_DIR/evaluation/REVOS/results.json" \
  --save_name "lang_inj_REVOS.json"
```

## Submission Mode

For datasets without a metric script in `evaluation.sh`, inference is launched
with `--submit` and the generated `Annotations/` folder is zipped.

General form:

```bash
bash projects/sa2va/evaluation/dist_test.sh \
  projects/sa2va/evaluation/sa2va_eval_ref_vos_cot_v2.py \
  "$MODEL_DIR" 8 \
  --work_dir "$MODEL_DIR/evaluation/$DATASET" \
  --dataset "$DATASET" \
  --max_turns 3 \
  --max_video_sample 20 \
  --max_temporal_frames_per_round 5 \
  --max_select_K 8 \
  --submit
```

Then:

```bash
cd "$MODEL_DIR/evaluation/$DATASET"
zip -qr "${DATASET}.zip" Annotations/
```
