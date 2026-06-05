import os
import json
import torch
import decord
import numpy as np
import copy
import random
from PIL import Image
from torch.utils.data import Dataset
from .base_eval_dataset import BaseEvalDataset

# 定义与训练一致的 Prompt 模板
VTG_QUESTIONS = [
    "The text query is '{class_name}', can you find the temporal range related to this query?",
]

class TVGDataset(BaseEvalDataset):
    def __init__(self,
                 image_folder,
                 expression_file,
                 sampled_frames=64, # 必须与训练时的 K 保持一致
    ):
        super().__init__()
        self.vid2metaid = self.json_file_preprocess(expression_file)
        self.image_folder = image_folder
        self.sampled_frames = sampled_frames

    def __len__(self):
        return len(self.vid2metaid)

    def real_len(self):
        return len(self.vid2metaid)

    def json_file_preprocess(self, expression_file):
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)
        return expression_datas

    def _read_video_decord(self, video_path):
        # 保持原本的读取方式：读取所有帧
        if not os.path.exists(video_path):
            print(f"Warning: Video does not exist: {video_path}")
            return None, 0
            
        vr = decord.VideoReader(video_path)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        # 稍微优化：不需要一次性转成 PIL List，太吃内存，只转需要的
        return vr, video_fps

    def __getitem__(self, index):
        video_obj_info = copy.deepcopy(self.vid2metaid[index])
        video_path = os.path.join(self.image_folder, video_obj_info.get("video", video_obj_info.get("video_path", "")))
        
        # 1. 获取视频 Reader
        vr, video_fps = self._read_video_decord(video_path)
        
        if vr is None:
            return None

        total_frames = len(vr)

        # =========================================================
        # 简化采样逻辑：等间隔采样 vs 全量采样
        # =========================================================
        if total_frames > self.sampled_frames:
            # 如果视频帧数超过设定值，进行等间隔采样 (linspace)
            # 保证覆盖从第0帧到最后一帧
            indices = np.linspace(0, total_frames - 1, self.sampled_frames, dtype=int)
        else:
            # 如果视频帧数不足，直接获取所有帧
            indices = np.arange(total_frames)

        # 确保索引有序且唯一（防止 np.linspace 在极短视频下产生重复）
        indices = sorted(list(set(indices)))

        # 2. 读取图像
        frames_arr = vr.get_batch(indices).asnumpy()
        sampled_images = [Image.fromarray(frame) for frame in frames_arr]
        
        # 3. 获取对应时间戳 (用于 inference 时将 token 映射回时间)
        sampled_times = [idx / video_fps for idx in indices]

        # 4. 构造文本 Prompt
        question = VTG_QUESTIONS[0].format(class_name=video_obj_info['query'])

        # 5. 返回结果
        data_dict = {
            'type': 'video',
            'video_id': video_obj_info.get('id', str(index)),
            'gt_start': video_obj_info.get('start_time', 0.0),
            'gt_end': video_obj_info.get('end_time', 0.0),
            'duration_sec': video_obj_info.get('duration', total_frames / video_fps),
            'exp': video_obj_info['query'],
            
            # 核心输入数据
            'images': sampled_images,       # List[PIL.Image]
            'sampled_times': sampled_times, # List[float]
            'text_prompt': question         # str
        }
        
        return data_dict

def collate_fn_filter_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0: return None
    return batch[0]