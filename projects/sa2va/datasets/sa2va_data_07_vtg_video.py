import os
import json
import random
import numpy as np
import torch
from typing import Literal, List, Dict, Any
from PIL import Image
import copy
import cv2

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Warning: decord not found. Please install it using `pip install decord`")

from projects.sa2va.datasets.base import Sa2VABaseDataset
from projects.sa2va.datasets.common import VTG_QUESTIONS

# 引入 Qwen3VL 特有的常量
IGNORE_INDEX = -100

class Sa2VA07VTGVideoDataset(Sa2VABaseDataset):

    def __init__(self,
                 image_folder,
                 expression_file,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=8192,
                 special_tokens=None,
                 arch_type: Literal['qwen'] = 'qwen', # 强制 qwen
                 preprocessor=None,
                 select_number=5,
                 sampled_frames=64, # 物理采样的帧数
                 dataset_type: Literal['default']='default',
                 extract_fps=1, # 逻辑帧率，用于定义 LLM 输出的坐标系
                 # 新增像素限制参数
                 min_pixels=4 * 28 * 28, # 调大一点，避免过小
                 max_pixels=32 * 28 * 28, # 限制每帧的最大像素数
                 **kwargs):
        
        # 确保传入了 Qwen3VL 专用的 processor
        if arch_type != 'qwen':
            raise ValueError("This dataset implementation strictly requires arch_type='qwen' for Qwen3-VL video encoding.")

        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            **kwargs
        )
        
        self.image_folder = image_folder
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        self.dataset_type = dataset_type
        self.extract_fps = extract_fps 
        self.video_data = self.load_jsonl(expression_file)

        # 像素限制
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Qwen3-VL 特有参数
        self.merge_size = 2 # Qwen3-VL 默认时间维度 merge 为 2
        
        # 获取 token id
        self.video_token = "<|video_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        
        # 确保 tokenizer 包含这些 token
        if self.tokenizer:
            self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token)

    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def real_len(self):
        return len(self.video_data)

    def _get_raw_video_frames(self, video_path):
        """
        读取视频原始帧，返回 list of numpy array，以及采样的时间戳。
        """
        if not os.path.exists(video_path):
            return None, None, None, None

        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            raw_total_frames = len(vr)
            raw_fps = vr.get_avg_fps()
            duration = raw_total_frames / raw_fps
            
            # 1. 物理采样
            if raw_total_frames >= self.sampled_frames:
                raw_indices = np.linspace(0, raw_total_frames - 1, self.sampled_frames, dtype=int)
            else:
                raw_indices = np.arange(raw_total_frames) 
            
            raw_indices = sorted(list(set(raw_indices)))
            
            # 读取像素数据 (T, H, W, C)
            frames_arr = vr.get_batch(raw_indices).asnumpy()
            
            # 转换为 List[np.ndarray] (H, W, C)
            frame_list = [f for f in frames_arr]

            # 获取每帧的物理时间戳
            frame_times = vr.get_frame_timestamp(raw_indices)[:, 0]
            
            return frame_list, raw_indices, frame_times, duration

        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None, None, None, None

    def _resize_image_to_limit(self, image, min_pixels, max_pixels):
        """
        使用 cv2 对 numpy image 进行 resize，保持长宽比，像素总数限制在 [min, max] 之间。
        Input: image (H, W, C) numpy array
        Output: image (H_new, W_new, C) numpy array
        """
        h, w = image.shape[:2]
        pixel_count = w * h
        
        if min_pixels <= pixel_count <= max_pixels:
            return image
            
        # 计算目标像素数
        if pixel_count < min_pixels:
            target_pixels = min_pixels
        else:
            target_pixels = max_pixels
            
        # 计算缩放比例
        ratio = (target_pixels / pixel_count) ** 0.5
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 确保最小尺寸 (Qwen 推荐 patch size 的倍数，这里简化处理，Processor 会二次处理)
        new_w = max(28, new_w)
        new_h = max(28, new_h)
        
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized_image

    def _sample_timestamps_to_grid(self, frame_times, grid_t):
        """
        从原始帧时间戳中采样出 grid_t 个时间戳，用于匹配模型的输出特征数。
        """
        if grid_t <= 0:
            return []
        
        # 简单均匀采样：从 len(frame_times) 中取 grid_t 个点
        indices = np.linspace(0, len(frame_times) - 1, grid_t)
        sampled_times = [frame_times[int(i)] for i in indices]
        return sampled_times

    def prepare_data(self, index):
        index = index % self.real_len()
        item = self.video_data[index]
        
        video_rel_path = item['video_path']
        video_path = os.path.join(self.image_folder, video_rel_path)
        
        # 1. 获取视频数据
        frame_list, raw_indices, frame_times, duration = self._get_raw_video_frames(video_path)
        
        if frame_list is None:
            return self.prepare_data(random.randint(0, self.real_len() - 1))

        # 2. 预处理：Resize (使用 CV2)
        frame_list = [self._resize_image_to_limit(v_img, self.min_pixels, self.max_pixels) for v_img in frame_list]

        # 3. 调用 Qwen3VL Processor
        try:
            # do_sample_frames=False 保证这里不再丢帧，特征数量由 frame_list 长度和内部 merge 决定
            video_inputs = self.preprocessor.video_processor(
                videos=frame_list, 
                return_metadata=True,
                do_sample_frames=False, 
            )
            
            pixel_values_videos = video_inputs['pixel_values_videos'] 
            video_grid_thw = video_inputs['video_grid_thw'] # shape: [1, 3] -> [[T_grid, H_grid, W_grid]]
            
        except Exception as e:
            print(f"Processor error at index {index}: {e}")
            return self.prepare_data(random.randint(0, self.real_len() - 1))

        # 4. 核心修复：根据 video_grid_thw 构建 Prompt
        # =========================================================
        grid = video_grid_thw[0] # [T_grid, H_grid, W_grid]
        grid_t = int(grid[0])
        grid_h = int(grid[1])
        grid_w = int(grid[2])
        
        # 重新采样时间戳以匹配 grid_t (Features 时间维度)
        merged_timestamps = self._sample_timestamps_to_grid(frame_times, grid_t)
        
        video_special_str = ""
        
        # 计算每帧需要的 token 数量
        # spatial_merge_size 通常为 2，意味着每 2x2 个 grid 对应 1 个 token
        spatial_merge_size = self.preprocessor.video_processor.merge_size
        spatial_tokens = (grid_h * grid_w) // (spatial_merge_size ** 2)
        
        # 循环 grid_t 次，确保 Token 数量与 Feature 时间维度严格一致
        for i in range(grid_t):
            curr_time = merged_timestamps[i]
            
            video_special_str += f"<{curr_time:.1f} seconds>"
            video_special_str += (
                self.vision_start_token + 
                self.video_token * spatial_tokens + 
                self.vision_end_token
            )
        # =========================================================

        # 5. 构建 VTG 任务对话
        dense_total_frames = int(duration * self.extract_fps)
        dense_indices = [int(t * self.extract_fps) for t in frame_times]
        
        events = item['events']
        if len(events) == 0: return None
        
        if len(events) >= self.select_number:
            selected_indexes = np.random.choice(len(events), self.select_number, replace=False)
        else:
            selected_indexes = np.random.choice(len(events), self.select_number, replace=True)
        selected_events = [events[_idx] for _idx in selected_indexes]

        conversations = []
        # addition_prompt = f"The video has total {dense_total_frames} frames. The sampled indices is {dense_indices}."
        addition_prompt=""
        
        for i, event in enumerate(selected_events):
            query = event['query']
            span = event['span'][0]
            start_time, end_time = span[0], span[1]
            
            start_frame_idx = int(start_time * self.extract_fps)
            end_frame_idx = int(end_time * self.extract_fps)
            start_frame_idx = min(max(start_frame_idx, 0), dense_total_frames - 1)
            end_frame_idx = min(max(end_frame_idx, 0), dense_total_frames - 1)
            if start_frame_idx > end_frame_idx: end_frame_idx = start_frame_idx
            
            answer_dict = {"start_frame": start_frame_idx, "end_frame": end_frame_idx}
            answer_str = json.dumps(answer_dict)
            question = random.choice(VTG_QUESTIONS).format(class_name=query)
            
            if i == 0:
                human_input = f"{video_special_str}\n{addition_prompt}\n{question}"
            else:
                human_input = query
            
            conversations.append({'from': 'human', 'value': human_input})
            conversations.append({'from': 'gpt', 'value': answer_str})

        # 6. Tokenization
        input_ids = []
        labels = []
        
        # System Prompt
        # system_text = "You are a helpful assistant."
        # system_content = f"<|im_start|>system\n{system_text}<|im_end|>\n"
        # system_ids = self.tokenizer.encode(system_content, add_special_tokens=False)
        # input_ids.extend(system_ids)
        # labels.extend([IGNORE_INDEX] * len(system_ids))
        
        for i in range(0, len(conversations), 2):
            human_item = conversations[i]
            gpt_item = conversations[i+1]
            
            human_text = human_item['value']
            gpt_text = gpt_item['value']
            
            human_text_formatted = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n"
            
            human_ids = self.tokenizer.encode(human_text_formatted, add_special_tokens=False)
            input_ids.extend(human_ids)
            labels.extend([IGNORE_INDEX] * len(human_ids))
            
            gpt_ids = self.tokenizer.encode(f"{gpt_text}<|im_end|>\n", add_special_tokens=False)
            input_ids.extend(gpt_ids)
            labels.extend(gpt_ids)

        # 7. Truncation & No Padding
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]
        
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)

        out_data_dict = {
            'input_ids': input_ids_tensor,
            'labels': labels_tensor,
            'attention_mask': attention_mask,
            'pixel_values': None,
            'image_grid_thw': None,
            'pixel_values_videos': pixel_values_videos,
            'video_grid_thw': video_grid_thw,
            "mask": None,
            'type': 'video',
            'g_pixel_values': None, # 如果模型不需要 global pixel values
        }
        
        return out_data_dict


if __name__ == '__main__':
    from transformers import AutoTokenizer, Qwen3VLProcessor
    from xtuner.utils import PROMPT_TEMPLATE
    from projects.sa2va.models import DirectResize
    
    dataset = Sa2VA07VTGVideoDataset(
        image_folder='data/VTG/TimeLens-100K/video',
        expression_file='data/VTG/TimeLens-100K/timelens-100k.jsonl',
        select_number=5,
        sampled_frames=64,
        extract_fps=1,  # 设定为 6 FPS
        dataset_type='default',
        arch_type='qwen',
        preprocessor=dict(
            type=Qwen3VLProcessor.from_pretrained,
            pretrained_model_name_or_path='pretrained/Qwen3-VL-4B-Instruct',
            trust_remote_code=True,
        ),
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path='pretrained/Qwen3-VL-4B-Instruct',
            trust_remote_code=True,
            padding_side='right'),
        special_tokens=['[SEG]', '<p>', '</p>', '<vp>', '</vp>'],
        extra_image_processor=dict(
                type=DirectResize,
                target_length=1024,
            ),
        prompt_template=PROMPT_TEMPLATE.qwen_chat,
        max_length=8192)
    

    # 运行测试
    # for i in range(10):
        # dataset.visualize_sample(i, output_dir='vis_debug_new_logic')
    dataset.prepare_data(0)