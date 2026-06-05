import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import os.path as osp
import time
import argparse
import json
import numpy as np
from collections import defaultdict
from PIL import Image, ImageFile
from pycocotools import mask as cocomask
import multiprocessing as mp
from tqdm import tqdm

from third_parts.revos.utils.metircs import db_eval_iou, db_eval_boundary

# 增加 PIL 对截断图片的处理容忍度
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 配置
NUM_WORKERS = 128
IGNORE_VIDEOS = ["71fb69bbcd"]

# 全局变量（用于子进程共享只读数据，避免重复 Pickle）
shared_meta = None
shared_preds = None
shared_gt_root = None

def init_worker(meta, preds, gt_root):
    """子进程初始化函数"""
    global shared_meta, shared_preds, shared_gt_root
    shared_meta = meta
    shared_preds = preds
    shared_gt_root = gt_root

def evaluate_single_task(task_args):
    """单个任务的评估逻辑"""
    vid_name, exp = task_args
    exp_name = f'{vid_name}_{exp}'

    try:
        # 获取共享数据
        vid_info = shared_meta.get(vid_name)
        pred_entry = shared_preds.get(vid_name, {}).get(exp)
        
        if not vid_info or not pred_entry:
            return None

        pred_rles = pred_entry.get('prediction_masks', [])
        if not pred_rles:
            return None

        # 基础信息准备
        h, w = pred_rles[0]['size']
        frames = vid_info['frames']
        vid_len = len(frames)
        num_rounds = len(pred_entry.get('text_prediction', []))
        exp_type = vid_info['expressions'][exp].get('type', 'Unknown')
        
        # 准备 Masks
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        
        # 获取目标对象 ID
        target_obj_ids = vid_info['expressions'][exp].get('obj_id', [])
        if not target_obj_ids:
            target_obj_ids = vid_info['expressions'][exp].get('anno_id', [])
        if isinstance(target_obj_ids, int):
            target_obj_ids = [target_obj_ids]

        # 1. 加载 GT 和 Prediction
        video_anno_path = os.path.join(shared_gt_root, "Annotations", vid_name)
        
        for frame_idx, frame_name in enumerate(frames):
            # 加载 GT
            for oid in target_obj_ids:
                gt_path = os.path.join(video_anno_path, str(oid), frame_name + ".png")
                if os.path.exists(gt_path):
                    try:
                        with Image.open(gt_path) as mask_img:
                            mask_img.load() 
                            mask_np = np.array(mask_img)
                            gt_masks[frame_idx] += (mask_np > 0).astype(np.uint8)
                    except Exception:
                        print(f"Error loading GT image: {gt_path}")
                        pass # 忽略损坏图片

            gt_masks[frame_idx] = (gt_masks[frame_idx] > 0).astype(np.uint8)
            
            # 加载 Pred
            if frame_idx < len(pred_rles):
                pred_masks[frame_idx] = cocomask.decode(pred_rles[frame_idx])

        # 2. 计算基础指标
        j_score = db_eval_iou(gt_masks, pred_masks).mean()
        f_score = db_eval_boundary(gt_masks, pred_masks).mean()

        # 3. 计算 Sample Index 指标
        j_sample = 0.0
        sample_indices = pred_entry.get('sample_index', [])
        if len(sample_indices) > 0:
            idx_arr = np.array(sample_indices, dtype=int)
            valid_indices = idx_arr[idx_arr < vid_len]
            if len(valid_indices) > 0:
                j_sample = db_eval_iou(gt_masks[valid_indices], pred_masks[valid_indices]).mean()

        # 4. 计算 Keyframe 指标 & 统计空 GT
        j_keyframe = 0.0
        n_k_empty = 0
        n_k_total = 0
        
        keyframe_indices = pred_entry.get('keyframe_index', [])
        if len(keyframe_indices) > 0:
            idx_arr_k = np.array(keyframe_indices, dtype=int)
            valid_indices_k = idx_arr_k[idx_arr_k < vid_len]
            n_k_total = len(valid_indices_k)
            
            if n_k_total > 0:
                # Keyframe IoU
                j_keyframe = db_eval_iou(gt_masks[valid_indices_k], pred_masks[valid_indices_k]).mean()
                
                # 统计 Keyframe 对应的 GT 是否为空
                for k_idx in valid_indices_k:
                    if not np.any(gt_masks[k_idx]):
                        n_k_empty += 1

        # 返回结果元组
        return {
            'key': exp_name,
            'vals': [j_score, f_score, j_sample, j_keyframe, num_rounds, vid_len, exp_type, n_k_empty, n_k_total]
        }

    except Exception as e:
        print(f"Error in {exp_name}: {e}")
        return None

def save_results(results_list, save_path):
    print(f"\nProcessing {len(results_list)} results for output...")
    
    output_dict = {}
    all_j, all_f, all_j_sample, all_j_keyframe = [], [], [], []
    
    round_stats = defaultdict(lambda: {'j': [], 'j_sample': [], 'j_keyframe': [], 'vid_len': []})
    type_stats = defaultdict(lambda: {'j': [], 'f': [], 'j_sample': [], 'j_keyframe': []})
    
    global_k_empty = 0
    global_k_total = 0

    for res in results_list:
        key = res['key']
        # Unpack values
        j, f, j_samp, j_key, n_rounds, v_len, e_type, n_k_empty, n_k_total = res['vals']
        
        output_dict[key] = res['vals'] # 保持原始格式用于写入 json
        
        all_j.append(j); all_f.append(f)
        all_j_sample.append(j_samp); all_j_keyframe.append(j_key)
        
        round_stats[n_rounds]['j'].append(j)
        round_stats[n_rounds]['j_sample'].append(j_samp)
        round_stats[n_rounds]['j_keyframe'].append(j_key)
        round_stats[n_rounds]['vid_len'].append(v_len)

        type_stats[e_type]['j'].append(j)
        type_stats[e_type]['f'].append(f)
        type_stats[e_type]['j_sample'].append(j_samp)
        type_stats[e_type]['j_keyframe'].append(j_key)
        
        global_k_empty += n_k_empty
        global_k_total += n_k_total

    if not all_j:
        print("No valid results found.")
        return

    # Calculate Keyframe Empty Ratio
    k_empty_ratio = 0.0
    if global_k_total > 0:
        k_empty_ratio = (global_k_empty / global_k_total) * 100

    final_json = {
        'Global': {
            'J': round(100 * float(np.mean(all_j)), 2),
            'F': round(100 * float(np.mean(all_f)), 2),
            'J&F': round(100 * float((np.mean(all_j) + np.mean(all_f)) / 2), 2),
            'J_sample': round(100 * float(np.mean(all_j_sample)), 2),
            'J_keyframe': round(100 * float(np.mean(all_j_keyframe)), 2),
            'Keyframe_GT_Empty_Ratio': round(k_empty_ratio, 2)
        },
        'Type_Statistics': {},
        'Round_Statistics': {}
    }

    # Print Global Stats
    print("\n" + "="*50)
    print(f"Keyframe Analysis (Total Keyframes: {global_k_total})")
    print("-" * 50)
    print(f"Empty GT Keyframes : {global_k_empty}")
    print(f"GT Empty Ratio     : {k_empty_ratio:.2f}%")
    print("="*50)
    
    # Process Stats Tables
    for r in sorted(round_stats.keys()):
        stats = round_stats[r]
        final_json['Round_Statistics'][r] = {
            'count': len(stats['j']),
            'J': round(100 * float(np.mean(stats['j'])), 2),
            'J_sample': round(100 * float(np.mean(stats['j_sample'])), 2),
            'J_keyframe': round(100 * float(np.mean(stats['j_keyframe'])), 2),
            'avg_vid_len': round(float(np.mean(stats['vid_len'])), 1)
        }
    
    for t in sorted(type_stats.keys()):
        stats = type_stats[t]
        final_json['Type_Statistics'][t] = {
            'count': len(stats['j']),
            'J&F': round(100 * float((np.mean(stats['j']) + np.mean(stats['f'])) / 2), 2),
            'J': round(100 * float(np.mean(stats['j'])), 2),
            'F': round(100 * float(np.mean(stats['f'])), 2),
            'J_sample': round(100 * float(np.mean(stats['j_sample'])), 2)
        }

    # Save
    with open(save_path, 'w') as f:
        json.dump(final_json, f, indent=4)
    print(f"\nResults successfully saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str)
    parser.add_argument("--meta_path", type=str, default="data/video_datas/Long-RVOS/valid/meta_expressions.json")
    parser.add_argument("--gt_root", type=str, default="data/video_datas/Long-RVOS/valid")
    parser.add_argument("--save_name", type=str, default="longrvos_eval_results.json")
    args = parser.parse_args()
    
    print(f"Loading metadata: {args.meta_path}")
    with open(args.meta_path, 'r') as f:
        meta_json = json.load(f)
        exp_dict = meta_json.get('videos', meta_json)

    print(f"Loading predictions: {args.pred_path}")
    with open(args.pred_path, 'r') as f:
        mask_pred = json.load(f)

    # 准备任务列表
    tasks = []
    skipped = 0
    for vid_name, vid_data in exp_dict.items():
        if vid_name in IGNORE_VIDEOS:
            skipped += 1
            continue
        
        if vid_name in mask_pred:
            for exp in vid_data['expressions']:
                if exp in mask_pred[vid_name]:
                    tasks.append((vid_name, exp))

    print(f"Tasks prepared: {len(tasks)} (Skipped Videos: {skipped})")
    print(f"Starting evaluation with {NUM_WORKERS} workers...")

    start_time = time.time()
    
    # 使用 Pool 进行并行计算
    # initializer 用于给每个子进程传递只读的元数据
    valid_results = []
    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(exp_dict, mask_pred, args.gt_root)) as pool:
        # 使用 imap_unordered 结合 tqdm 显示进度
        for res in tqdm(pool.imap_unordered(evaluate_single_task, tasks), total=len(tasks)):
            if res is not None:
                valid_results.append(res)
    
    # 汇总保存
    output_path = osp.join(osp.dirname(args.pred_path), args.save_name)
    save_results(valid_results, output_path)
    
    print(f"Total Time: {time.time() - start_time:.2f}s")