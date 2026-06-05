import argparse
from collections import defaultdict
import json
import os
import mmengine
import numpy as np
from PIL import Image
import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoTokenizer, AutoProcessor
import concurrent.futures
from pycocotools import mask as cocomask

from projects.sa2va.hf.models_qwen3vl.modeling_sa2va_qwen_cotv3_keyframe_withvis import Sa2VAChatModelQwenCOT
from projects.sa2va.evaluation.dataset import RefVOSDatasetCOT_v2
from projects.sa2va.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu


DATASETS_INFO = {
    'DAVIS': {
        'data_root': 'data/video_datas/davis17/',
        'image_folder': 'data/video_datas/davis17/valid/JPEGImages/',
        'expression_file': 'data/video_datas/davis17/meta_expressions/valid/meta_expressions.json',
        'mask_file': 'data/video_datas/davis17/valid/mask_dict.pkl',
    },
    'MEVIS': {
        'data_root': 'data/video_datas/mevis/valid/',
        'image_folder': 'data/video_datas/mevis/valid/JPEGImages',
        'expression_file': 'data/video_datas/mevis/valid/meta_expressions.json',
        'mask_file': None,
    },
    'MEVIS_U': {
        'data_root': 'data/video_datas/mevis/valid_u/',
        'image_folder': 'data/video_datas/mevis/valid_u/JPEGImages',
        'expression_file': 'data/video_datas/mevis/valid_u/meta_expressions.json',
        'mask_file': 'data/video_datas/mevis/valid_u/mask_dict.json',
    },
    'MEVIS_T': {
        'data_root': 'data/video_datas/mevis/test/',
        'image_folder': 'data/video_datas/mevis/test/JPEGImages',
        'expression_file': 'data/video_datas/mevis/test/meta_expressions_release.json',
        'mask_file': None,
    },
    'REFYTVOS': {
        'data_root': 'data/video_datas/rvos/',
        'image_folder': 'data/video_datas/rvos/valid/JPEGImages/',
        'expression_file': 'data/video_datas/rvos/meta_expressions/valid/meta_expressions.json',
        'mask_file': None,
    },
    'REVOS': {
        'data_root': 'data/video_datas/revos/',
        'image_folder': 'data/video_datas/revos/',
        'expression_file': 'data/video_datas/revos/meta_expressions_valid_.json',
        'mask_file': None,
    },
    'REF_SAV': {
        'data_root': 'data/video_datas/ref_sav_eval',
        'image_folder': 'data/video_datas/ref_sav_eval/videos',
        'expression_file': 'data/video_datas/ref_sav_eval/meta_expressions_valid.json',
        'mask_file': 'data/video_datas/ref_sav_eval/mask_dict.json',
    },
    'LONGRVOS': {
        'data_root': 'data/video_datas/Long-RVOS/valid',  
        'image_folder': 'data/video_datas/Long-RVOS/valid/JPEGImages',
        'expression_file': 'data/video_datas/Long-RVOS/valid/meta_expressions.json',
        'mask_file': None, 
    },
    'REASONVOS': {
        'data_root': 'data/video_datas/reasonvos',  
        'image_folder': 'data/video_datas/reasonvos/JPEGImages',
        'expression_file': 'data/video_datas/reasonvos/meta_expressions_format.json',
        'mask_file': None, 
    },
    'GROUNDMORE': {
        'data_root': 'data/video_datas/GroundMoRe',  
        'image_folder': 'data/video_datas/GroundMoRe/annotations',
        'expression_file': 'data/video_datas/GroundMoRe/test_v2.json',
        'mask_file': None, 
    },
}


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def mask_save(item, mask_prediction, work_dir):
    vid_id = item['video_id']
    exp_id = item['exp_id']
    save_path = os.path.join(work_dir, 'Annotations', vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert('L')
        file_name = item['frames'][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)

def parse_args():
    parser = argparse.ArgumentParser(description='RefVOS')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument('--dataset', default='MEVIS_U', help='Specify a dataset')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--data_root', default='./data', help='Root directory')
    parser.add_argument('--max_turns', type=int, default=3)
    parser.add_argument('--max_video_sample', type=int, default=0)
    parser.add_argument('--max_select_K', type=int, default=5)
    parser.add_argument('--max_temporal_frames_per_round', type=int, default=10)
    parser.add_argument('--vis_save_path', type=str, default=None)
    parser.add_argument('--lang_embed_injecting_per_frame', action='store_true', default=True)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

if __name__ == '__main__':
    args = parse_args()

    # Dataset path correction
    for key, info in DATASETS_INFO.items():
        for path_key, path_val in info.items():
            if path_val is not None and ('folder' in path_key or 'file' in path_key or 'root' in path_key):
                DATASETS_INFO[key][path_key] = os.path.join(args.data_root, os.path.relpath(path_val, './data'))

    work_dir = args.work_dir if args.work_dir else 'work_dirs/default'

    # DDP Initialization
    if args.launcher == 'none':
        rank = 0
        world_size = 1
    elif args.launcher == 'pytorch':
        import datetime
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
    elif args.launcher == 'slurm':
        _init_dist_slurm('nccl')
        rank, world_size = get_dist_info()

    # Load Model
    model = Sa2VAChatModelQwenCOT.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    dataset_info = DATASETS_INFO[args.dataset]
    dataset = RefVOSDatasetCOT_v2(
        image_folder=dataset_info['image_folder'],
        expression_file=dataset_info['expression_file'],
        mask_file=dataset_info['mask_file'],
    )

    model.max_video_sample = args.max_video_sample
    model.max_select_K = args.max_select_K
    dataset.max_video_sample = args.max_video_sample
    dataset.max_select_K = args.max_select_K

    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=8, 
        pin_memory=True, 
        collate_fn=lambda x:x[0],
    )

    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    
    # 注意：这里的 round_stat 是本地的，后面需要汇总
    # local_round_stat = defaultdict(int) 

    # Main Loop
    for item in tqdm.tqdm(dataloader, disable=(rank!=0)):
        with torch.no_grad():
            result = model.predict_forward(
                processed_sam2_images=item['processed_sam2_images'],
                qwen_video_frames=item['qwen_video_frames'],
                qwen_spatial_frames=item['qwen_spatial_frames'],
                video_sample_index=item['video_sample_index'],
                image_sample_index=item['image_sample_index'],
                video=item['images'],
                text=item['text_prompt'],
                tokenizer=tokenizer,
                processor=processor,
                max_turns=args.max_turns,
                max_temporal_frames_per_round=args.max_temporal_frames_per_round,
                vis_save_path=args.vis_save_path,
                sample_id=f"{item['video_id']}_{item['exp_id']}",
                lang_embed_injecting_per_frame=args.lang_embed_injecting_per_frame
            )

        # Result Parsing
        text_prediction = []
        # round = 0 # 局部变量，容易和内置函数混淆
        assistant_rounds = 0
        for cur_round in result['prediction']:
            if cur_round["role"] == "assistant":
                assistant_rounds += 1
                text_prediction.append(cur_round["content"])
        
        # 本地统计可以保留，但主要依赖最后的全局汇总
        # local_round_stat[str(assistant_rounds)] += 1
        
        mask_prediction = result['prediction_masks'][0] if len(result['prediction_masks']) > 0 else np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)
        prediction_masks_shot = result['prediction_masks_shot'] if len(result.get('prediction_masks_shot', [])) > 0 else np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)

        if args.submit:
            async_func(executor, mask_save, item=item, mask_prediction=mask_prediction, work_dir=work_dir)
            encoded_mask = None
        else:
            encoded_mask = mask_to_rle(mask_prediction)

        res_entry = {
            'index': item['index'],
            'video_id': item['video_id'],
            'exp_id': item['exp_id'],
            'text_prediction': text_prediction,
            'num_rounds': assistant_rounds, # <--- 新增：每个样本记录自己的轮数
            'frames': item['frames'],
            'exp': item['text_prompt'], 
            'prediction_masks': encoded_mask,
            'prediction_masks_shot': mask_to_rle(prediction_masks_shot), 
            "sample_index": result["sample_index"].tolist() if isinstance(result["sample_index"], np.ndarray) else result["sample_index"],
            "keyframe_index": result["keyframe_index"].tolist() if isinstance(result["keyframe_index"], np.ndarray) else result["keyframe_index"]
        }

        results.append(res_entry)

    executor.shutdown(wait=True)
    print(f'[Rank {rank}] : Finished.')
    
    # 汇总结果
    if not args.submit:
        # 收集所有卡的数据
        tmpdir = os.path.join(work_dir, "tmp_data")
        results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
        
        if rank == 0:
            final_results = {}
            global_round_stat = defaultdict(int)

            for item in results:
                vid_id = item['video_id']
                exp_id = item['exp_id']
                
                # 直接使用前面存好的 num_rounds
                n_rounds = item.get('num_rounds', 0)
                global_round_stat[n_rounds] += 1

                if vid_id not in final_results:
                    final_results[vid_id] = {}
                final_results[vid_id][exp_id] = item
            
            # 1. 保存原始结果 results.json
            mmengine.mkdir_or_exist(work_dir)
            json.dump(final_results, open(f'{work_dir}/results.json', 'w'))
            
            # 2. 准备并保存轮数统计 round_stats.json
            print("\n" + "="*40)
            print(f"Global Round Statistics (Total Samples: {len(results)}):")
            
            stats_to_save = {
                "total_samples": len(results),
                "round_details": {}
            }
            
            for r in sorted(global_round_stat.keys()):
                count = global_round_stat[r]
                percentage = (count / len(results)) * 100
                
                # 打印到控制台
                print(f"  Round {r}: {count} samples ({percentage:.2f}%)")
                
                # 记录到字典
                stats_to_save["round_details"][str(r)] = {
                    "count": count,
                    "percentage": f"{percentage:.2f}%"
                }
            print("="*40 + "\n")
            
            # 保存统计文件
            stats_path = os.path.join(work_dir, 'round_stats.json')
            json.dump(stats_to_save, open(stats_path, 'w'), indent=4)
            print(f"Statistics saved to {stats_path}")
            
            print('Done')