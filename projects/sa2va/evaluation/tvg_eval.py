import argparse
import json
import os
import tqdm
import torch
import numpy as np
import mmengine
from mmengine.dist import get_dist_info, init_dist
from torch.utils.data import DistributedSampler

from transformers import AutoTokenizer, Qwen3VLProcessor
from projects.sa2va.hf.models_qwen3vl.modeling_sa2va_qwen_cotv3_keyframe_withvis import Sa2VAChatModelQwenCOT
from projects.sa2va.evaluation.dataset.TVG import TVGDataset, collate_fn_filter_none

# Dataset Configuration Table
DATASETS_INFO = {
    'CHARADES': {
        'image_folder': 'data/VTG_data/videos_1FPS', 
        'expression_file': 'data/VTG_data/charades_test.json'
    },
    'ActivityNet': {
        'image_folder': 'data/VTG_data/videos_1FPS',
        'expression_file': 'data/VTG_data/activitynet_val_2_test.json'
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description='VTG Evaluation Multi-GPU No-Part-Files')
    parser.add_argument('model_path', help='HuggingFace model path.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='Job launcher')
    parser.add_argument('--dataset', default='CHARADES', help='Dataset name')
    parser.add_argument('--work_dir', type=str, default='./work_dirs/eval_vtg')
    parser.add_argument('--sampled_frames', type=int, default=32, help='Number of frames to sample (K)')
    parser.add_argument('--max_pixel', type=int, default=64, help='Max pixels factor (N*28*28)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Distributed Initialization
    if args.launcher == 'none':
        rank = 0
        world_size = 1
    else:
        init_dist(args.launcher)
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # Create working directory (Rank 0 only)
    if rank == 0:
        mmengine.mkdir_or_exist(args.work_dir)

    # 2. Load Model
    # low_cpu_mem_usage=True is crucial for multi-GPU loading to prevent OOM
    model = Sa2VAChatModelQwenCOT.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    
    processor = Qwen3VLProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.processor = processor

    # Set temporal pixel limit based on args
    if hasattr(model, 'max_pixel_temporal'):
        model.max_pixel_temporal = args.max_pixel * 28 * 28

    # 3. Load Dataset
    info = DATASETS_INFO.get(args.dataset, None)
    if info is None:
        raise ValueError(f"Dataset {args.dataset} not found in DATASETS_INFO.")
        
    dataset = TVGDataset(
        image_folder=info['image_folder'],
        expression_file=info['expression_file'],
        sampled_frames=args.sampled_frames
    )

    # 4. Create Distributed Sampler and DataLoader
    if world_size > 1:
        # shuffle=False is critical for reproducibility and deterministic ordering
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        sampler=sampler, 
        shuffle=False, 
        collate_fn=collate_fn_filter_none,
    )

    # 5. Inference Loop
    local_results = []
    
    if rank == 0:
        iterator = tqdm.tqdm(dataloader, desc=f"Inference (Rank {rank})")
    else:
        iterator = dataloader

    for item in iterator:        
        with torch.no_grad():
            pred_res = model.predict_vtg_video(
                video=item['images'],           
                text_prompt=item['text_prompt'],
                sample_times=item['sampled_times']
            )

        s_idx = pred_res.get('pred_start_idx', 0)
        e_idx = pred_res.get('pred_end_idx', len(item['images']))
        
        result_item = {
            'video_id': item['video_id'],
            'query': item['exp'],
            'gt_start': item['gt_start'],
            'gt_end': item['gt_end'],
            'pred_start': s_idx,
            'pred_end': e_idx,
            'duration': item['duration_sec'],
            'raw_output': pred_res['prediction_raw']
        }
        local_results.append(result_item)
            
    # 6. Gather Results from All Ranks
    if world_size > 1:
        torch.distributed.barrier() # Sync before gathering
        
        # Container to hold results from all ranks
        gather_output = [None for _ in range(world_size)]
        
        # Collect local_results from all ranks into gather_output
        torch.distributed.all_gather_object(gather_output, local_results)
    else:
        gather_output = [local_results]

    # 7. Merge, Deduplicate, and Save (Rank 0 Only)
    if rank == 0:
        final_results = []
        unique_keys = set()
        
        # Flatten list and deduplicate
        for rank_res in gather_output:
            for item in rank_res:
                # Create a unique key to handle padding from DistributedSampler
                # Combining video_id and query is usually sufficient for VTG tasks
                uid = f"{item['video_id']}_{item['query']}"
                
                if uid not in unique_keys:
                    unique_keys.add(uid)
                    final_results.append(item)
        target_dir = os.path.join(args.work_dir, f'{args.dataset}')
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, 'results.json')
        
        with open(save_path, 'w') as f:
            json.dump(final_results, f, indent=4)
            
        print(f"✅ Inference Finished. Collected {len(final_results)} unique samples.")
        print(f"✅ Results saved to: {save_path}")

if __name__ == '__main__':
    main()