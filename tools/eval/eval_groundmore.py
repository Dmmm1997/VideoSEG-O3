import argparse
import json
import math
import multiprocessing as mp
import os
import re
import warnings
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask
from skimage.morphology import disk

# Configuration
NUM_WORKERS = 128  # Adjust based on CPU cores
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------
# Helper Functions (Metrics)
# --------------------------

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index. """
    assert annotation.shape == segmentation.shape
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j

def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res

def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)
    return F

def _seg2bmap(seg, width=None, height=None):
    seg = seg.astype(bool)
    seg[seg > 0] = 1
    assert np.atleast_3d(seg).shape[2] == 1
    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height
    h, w = seg.shape[:2]
    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)
    assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1
    return bmap

def db_statistics(per_frame_values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)
    return M, O, None

def time_str_to_seconds(time_str):
    parts = time_str.split(":")
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

# --------------------------
# Main Evaluation Worker
# --------------------------

def eval_queue(q, out_dict, video_root):
    # Retrieve shared dictionaries from globals
    # Note: In 'spawn' method (Windows/MacOS), globals aren't shared implicitly. 
    # For Linux (fork), this works. Ideally, pass dicts or use an initializer.
    local_metadata = metadata_dict
    local_preds = pred_dict

    while not q.empty():
        try:
            video_id, exp_idx = q.get(timeout=1)
        except:
            break

        try:
            # --- 1. Parse Metadata (GroundMore Logic) ---
            expressions = local_metadata[video_id]["questions"]
            expression_list = list(expressions.keys())
            exp_id = expression_list[exp_idx]
            
            exp_data = expressions[exp_id]
            q_type = exp_data["q_type"]
            obj_id = int(exp_data["obj_id"])
            
            # Timestamp parsing specific to GroundMore
            clip_start = video_id[-9:].split("_")[0][:2] + ":" + video_id[-9:].split("_")[0][2:]
            start = exp_data["action_start"]
            end = exp_data["action_end"]
            action_start = (time_str_to_seconds(start) - time_str_to_seconds(clip_start)) * 6
            action_end = (time_str_to_seconds(end) - time_str_to_seconds(clip_start)) * 6 - 1

            # --- 2. Parse Prediction (ReasonVOS Logic) ---
            if video_id not in local_preds or exp_id not in local_preds[video_id]:
                print(f"Missing prediction: {video_id}/{exp_id}")
                continue
                
            pred_entry = local_preds[video_id][exp_id]
            pred_rles = pred_entry['prediction_masks']
            num_rounds = len(pred_entry.get('text_prediction', [''])) # Get round count
            
            # Frame setup
            total_frames = len(pred_rles)
            sample_indices = np.linspace(0, total_frames - 1, num=20, dtype=int)
            
            # --- 3. Construct Masks ---
            # GroundMore loads GT from disk, Pred from RLE
            gt_masks_list = []
            pred_masks_list = []
            
            # We need dimensions to create empty masks
            temp_mask = cocomask.decode(pred_rles[0])
            h, w = temp_mask.shape

            mask_dir = os.path.join(video_root, video_id, "masks/")
            
            for index in sample_indices:
                # A. Load GT
                if action_start <= index <= action_end:
                    # GroundMore usually has frame_000000.png
                    # We check both 6-digit and 7-digit conventions
                    mask_name_6 = f"frame_{str(index).zfill(6)}.png"
                    mask_name_7 = f"{str(index).zfill(7)}.png"
                    
                    mask_path = os.path.join(mask_dir, mask_name_6)
                    if not os.path.exists(mask_path):
                         mask_path = os.path.join(mask_dir, mask_name_7)

                    if os.path.exists(mask_path):
                        raw_mask = np.array(Image.open(mask_path).convert('P'))
                        gt_mask = (raw_mask == obj_id).astype(np.uint8)
                    else:
                        gt_mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    gt_mask = np.zeros((h, w), dtype=np.uint8)
                
                gt_masks_list.append(gt_mask)

                # B. Load Prediction
                p_mask = cocomask.decode(pred_rles[index])
                pred_masks_list.append(p_mask)

            gt_masks = np.stack(gt_masks_list, axis=0)
            pred_masks = np.stack(pred_masks_list, axis=0)
            
            # --- 4. Compute Metrics ---
            j_metric = db_eval_iou(gt_masks, pred_masks)
            f_metric = db_eval_boundary(gt_masks, pred_masks)
            
            JM, _, _ = db_statistics(j_metric)
            FM, _, _ = db_statistics(f_metric)

            out_dict[f'{video_id}_{exp_id}'] = {
                'J': JM, 
                'F': FM, 
                'q_type': q_type, 
                'rounds': num_rounds,
                'video': video_id
            }

        except Exception as e:
            print(f"Error processing {video_id} - {exp_idx}: {e}")

# --------------------------
# Main Entry Point
# --------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path', help="Path to prediction JSON file (ReasonVOS format)")
    parser.add_argument('--video_root', default="data/video_datas/GroundMoRe/annotations", help="Path to GroundMore annotations/images")
    parser.add_argument('--meta_file', default="data/video_datas/GroundMoRe/test_v2.json", help="Path to GroundMore metadata")
    parser.add_argument('--save_name', default="groundmore_eval_result.json")
    args = parser.parse_args()

    # Load Data
    print(f"Loading metadata from {args.meta_file}...")
    with open(args.meta_file, "r") as f:
        full_meta = json.load(f)
        metadata = full_meta["videos"] if "videos" in full_meta else full_meta

    print(f"Loading predictions from {args.pred_path}...")
    with open(args.pred_path, "r") as f:
        predictions = json.load(f)

    # Initialize MP resources
    manager = mp.Manager()
    queue = manager.Queue()
    output_dict = manager.dict()
    
    # Shared dicts (Read-only for workers)
    global metadata_dict, pred_dict
    metadata_dict = manager.dict(metadata)
    pred_dict = manager.dict(predictions)

    # Populate Queue
    cnt = 0
    for video in metadata:
        expressions = metadata[video]["questions"]
        num_expressions = len(expressions)
        for exp_idx in range(num_expressions):
            queue.put([video, exp_idx])
            cnt += 1
            
    print(f"Total expressions to evaluate: {cnt}")

    # Start Workers
    processes = []
    for rank in range(NUM_WORKERS):
        p = mp.Process(target=eval_queue, args=(queue, output_dict, args.video_root))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # --------------------------
    # Aggregation & Reporting
    # --------------------------
    results = list(output_dict.values())
    if not results:
        print("No results computed.")
        exit()

    # 1. Aggregate by Question Type (GroundMore Standard)
    type_stats = defaultdict(lambda: {'J': [], 'F': []})
    global_stats = {'J': [], 'F': []}
    
    # 2. Aggregate by Round (User Request)
    round_stats = defaultdict(lambda: {'J': [], 'F': [], 'count': 0})

    for res in results:
        j, f = res['J'], res['F']
        qt = res['q_type']
        rnd = res['rounds']
        
        # Type
        type_stats[qt]['J'].append(j)
        type_stats[qt]['F'].append(f)
        
        # Global
        global_stats['J'].append(j)
        global_stats['F'].append(f)
        
        # Round
        round_stats[rnd]['J'].append(j)
        round_stats[rnd]['F'].append(f)
        round_stats[rnd]['count'] += 1

    # --- Print Type Table ---
    print("\n" + "="*50)
    print(f"{'Question Type':<20} | {'J':<6} | {'F':<6} | {'J&F':<6}")
    print("-" * 50)
    
    final_output = {
        'Global': {},
        'Type_Statistics': {},
        'Round_Statistics': {}
    }

    # Process Types
    for qt in sorted(type_stats.keys()):
        mj = np.mean(type_stats[qt]['J'])
        mf = np.mean(type_stats[qt]['F'])
        mjf = (mj + mf) / 2
        print(f"{qt:<20} | {mj:.4f} | {mf:.4f} | {mjf:.4f}")
        final_output['Type_Statistics'][qt] = {'J': mj, 'F': mf, 'J&F': mjf}

    # Process Global
    g_j = np.mean(global_stats['J'])
    g_f = np.mean(global_stats['F'])
    g_jf = (g_j + g_f) / 2
    print("-" * 50)
    print(f"{'Global':<20} | {g_j:.4f} | {g_f:.4f} | {g_jf:.4f}")
    print("="*50 + "\n")
    
    final_output['Global'] = {'J': g_j, 'F': g_f, 'J&F': g_jf}

    # --- Print Round Table ---
    print("="*50)
    print(f"{'Round':<6} | {'Count':<6} | {'J':<6} | {'F':<6} | {'J&F':<6}")
    print("-" * 50)
    
    for r in sorted(round_stats.keys()):
        stats = round_stats[r]
        mj = np.mean(stats['J'])
        mf = np.mean(stats['F'])
        mjf = (mj + mf) / 2
        count = stats['count']
        print(f"{r:<6} | {count:<6} | {mj:.4f} | {mf:.4f} | {mjf:.4f}")
        final_output['Round_Statistics'][r] = {'J': mj, 'F': mf, 'J&F': mjf, 'count': count}
    print("="*50 + "\n")

    # Save
    output_path = os.path.join(os.path.dirname(args.pred_path), args.save_name)
    with open(output_path, 'w') as f:  # 注意这里，'w' 是 open 的第二个参数
        json.dump(final_output, f, indent=4)
    print(f"Results saved to {output_path}")