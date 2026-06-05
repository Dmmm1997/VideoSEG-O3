import os
import json
import random
import numpy as np
import torch
from typing import Literal
from PIL import Image
import copy
import cv2

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Warning: decord not found. Please install it using `pip install decord`")

from projects.sa2va.datasets.base import Sa2VABaseDataset
from projects.sa2va.datasets.common import VTG_QUESTIONS

class Sa2VA07VTGDataset(Sa2VABaseDataset):

    def __init__(self,
                 image_folder,
                 expression_file,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 select_number=5,
                 sampled_frames=64,
                 dataset_type: Literal['default']='default',
                 extract_fps=1, # 逻辑帧率，用于定义 LLM 的坐标系
                 **kwargs):
        
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            **kwargs
        )
        
        self.image_folder = image_folder
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        self.dataset_type = dataset_type
        self.extract_fps = extract_fps 

        self.min_pixels_multi = 4 * 28 * 28
        self.max_pixels_multi = 64 * 28 * 28

        self.video_data = self.load_jsonl(expression_file)

    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def real_len(self):
        return len(self.video_data)

    def _get_video_frames(self, video_path):
        """
        修正逻辑：
        1. 对原始视频进行全长等间隔采样，得到 sampled_frames 张图片。
        2. 计算这些图片对应在 extract_fps 坐标系下的 dense_indices。
        3. 计算视频在 extract_fps 下的 dense_total_frames。
        """
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return None, None, None
            
        try:
            # 使用 num_threads=1 避免某些环境下的死锁
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            raw_total_frames = len(vr)
            raw_fps = vr.get_avg_fps()
            duration = raw_total_frames / raw_fps
            
            # --- 1. 物理采样：全视频等间隔 ---
            if raw_total_frames >= self.sampled_frames:
                raw_indices = np.linspace(0, raw_total_frames - 1, self.sampled_frames, dtype=int)
            else:
                raw_indices = np.arange(raw_total_frames)
            
            raw_indices = sorted(list(set(raw_indices))) # 去重并排序

            # --- 2. 获取图像和时间戳 ---
            # get_batch 是解码的关键步骤
            frames_arr = vr.get_batch(raw_indices).asnumpy()
            images = [Image.fromarray(f) for f in frames_arr]
            
            # 获取这些帧的真实时间戳 (seconds)
            # get_frame_timestamp 返回 [[start, end], ...]，我们取 start
            frame_times = vr.get_frame_timestamp(raw_indices)[:, 0]
            
            # --- 3. 逻辑映射：转换到 extract_fps 坐标系 ---
            # 这里的 index 是 LLM 理解的 "第几帧"
            # 例如：时间 1.5s，extract_fps=6 -> index = int(1.5 * 6) = 9
            dense_indices = [int(t * self.extract_fps) for t in frame_times]
            
            # 计算逻辑总帧数 (LLM 认为视频有多少帧)
            dense_total_frames = int(duration * self.extract_fps)
            
            # 边界修正：防止浮点误差导致 index >= total
            dense_indices = [min(idx, dense_total_frames - 1) for idx in dense_indices]
            
            return images, dense_indices, dense_total_frames
            
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None, None, None

    def prepare_data(self, index):
        index = index % self.real_len()
        item = self.video_data[index]
        
        video_rel_path = item['video_path']
        video_path = os.path.join(self.image_folder, video_rel_path)
        
        # 获取图像、逻辑索引、逻辑总帧数
        images, dense_indices, dense_total_frames = self._get_video_frames(video_path)
        
        if images is None:
            new_index = random.randint(0, self.real_len() - 1)
            return self.prepare_data(new_index)
            
        events = item['events']
        
        if len(events) == 0:
            return None
        
        # 随机选择事件
        if len(events) >= self.select_number:
            selected_indexes = np.random.choice(len(events), self.select_number, replace=False)
        else:
            selected_indexes = np.random.choice(len(events), self.select_number, replace=True)
        
        selected_events = [events[_idx] for _idx in selected_indexes]
            
        conversations = []
        
        for i, event in enumerate(selected_events):
            query = event['query']
            span = event['span'][0]
            start_time, end_time = span[0], span[1]
            
            # === Label 转换 ===
            # 将真实时间转为逻辑帧索引
            start_frame_idx = int(start_time * self.extract_fps)
            end_frame_idx = int(end_time * self.extract_fps)
            
            # 边界保护
            start_frame_idx = min(max(start_frame_idx, 0), dense_total_frames - 1)
            end_frame_idx = min(max(end_frame_idx, 0), dense_total_frames - 1)
            
            if start_frame_idx > end_frame_idx:
                end_frame_idx = start_frame_idx
                
            answer_dict = {
                "start_frame": start_frame_idx,
                "end_frame": end_frame_idx
            }
            answer_str = json.dumps(answer_dict)
            
            question = random.choice(VTG_QUESTIONS).format(class_name=query)
            
            # === Prompt 注入 ===
            # 告诉 LLM：这个视频逻辑上有 dense_total_frames 帧
            # 你看到的这几张图，分别对应逻辑上的 dense_indices
            addition_prompt = f"The video has total {dense_total_frames} frames. The sampled indices is {dense_indices}."
            
            if i == 0:
                human_input = f"<image>\n{addition_prompt}\n{question}"
            else:
                human_input = query
                
            conversations.append({'from': 'human', 'value': human_input})
            conversations.append({'from': 'gpt', 'value': answer_str})

        out_data_dict = {}
        try:
            image_data = self._process_multiple_images(images)
            out_data_dict.update(image_data)
            
            num_frames = len(images)
            image_token_str = self._create_token_string(image_data['num_image_tokens'], num_frames)
            
            conversations_encoded = self._process_conversations_for_encoding(
                conversations, image_token_str, is_video=True
            )
            
            if self.arch_type == 'qwen' and 'num_frame_tokens' in image_data:
                 conversations_encoded = self._expand_video_tokens(
                    conversations_encoded, image_data['num_frame_tokens'], image_data['num_image_tokens']
                )

            token_dict = self.get_inputid_labels(conversations_encoded)
            out_data_dict.update(token_dict)
            out_data_dict['masks'] = None 
            out_data_dict['type'] = 'video'
            
            return out_data_dict

        except Exception as e:
            print(f"Error processing data at index {index}: {e}", flush=True)
            return None

    def visualize_sample(self, index, output_dir='vis_outputs'):
        """可视化验证"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        index = index % self.real_len()
        item = self.video_data[index]
        video_rel_path = item['video_path']
        video_path = os.path.join(self.image_folder, video_rel_path)
        
        # 调用新的接口
        images, dense_indices, dense_total_frames = self._get_video_frames(video_path)
        if images is None: return

        print(f"\n--- Visualizing Index {index} ---")
        print(f"Original Duration: {item['duration']}s")
        print(f"Extract FPS (Logical): {self.extract_fps}")
        print(f"Logical Total Frames: {dense_total_frames}")
        print(f"Sampled {len(images)} images")
        print(f"Logical Indices mapped: {dense_indices}")
        
        event = random.choice(item['events'])
        query = event['query']
        span = event['span'][0]
        start_time, end_time = span[0], span[1]

        # 验证 Label 转换
        start_label_idx = int(start_time * self.extract_fps)
        end_label_idx = int(end_time * self.extract_fps)
        
        print(f"Event: {query} ({start_time}-{end_time}s) -> Logical Label: [{start_label_idx}, {end_label_idx}]")

        clean_query = "".join([c if c.isalnum() else "_" for c in query])
        save_path = os.path.join(output_dir, f"id{index}_{clean_query}.webm")
        w, h = images[0].size
        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'VP80'), 5, (w, h))

        # 在图片上画出它是属于逻辑时间轴的哪一部分
        for i, img in enumerate(images):
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # 判断当前帧的逻辑 index 是否在 GT 范围内
            curr_dense_idx = dense_indices[i]
            is_in_gt = start_label_idx <= curr_dense_idx <= end_label_idx
            
            if is_in_gt:
                cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 255, 0), 10)
                cv2.putText(frame, "GT", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            info_text = f"ImgIdx:{i} | LogicalIdx:{curr_dense_idx}/{dense_total_frames}"
            cv2.putText(frame, info_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            video_writer.write(frame)

        video_writer.release()
        print(f"Saved visualization to: {save_path}")

if __name__ == '__main__':
    from transformers import AutoTokenizer, Qwen3VLProcessor
    from xtuner.utils import PROMPT_TEMPLATE
    from projects.sa2va.models import DirectResize
    
    dataset = Sa2VA07VTGDataset(
        image_folder='data/VTG/TimeLens-100K/video',
        expression_file='data/VTG/TimeLens-100K/timelens-100k.jsonl',
        select_number=5,
        sampled_frames=32,
        extract_fps=10,  # 设定为 6 FPS
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
    #     dataset.visualize_sample(i, output_dir='vis_debug_new_logic')
    dataset.prepare_data(0)