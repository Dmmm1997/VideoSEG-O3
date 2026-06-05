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
from qwen_vl_utils import process_vision_info

class Sa2VA07VTGDataset(Sa2VABaseDataset):

    def __init__(self,
                 image_folder,
                 expression_file,
                 prompt_template=None, # Not used directly, we use processor's template
                 tokenizer=None,
                 max_length=2048,
                 special_tokens=None,
                 arch_type: Literal['qwen'] = 'qwen',
                 preprocessor=None, # Must be Qwen3VLProcessor
                 select_number=5,
                 sampled_frames=64,
                 dataset_type: Literal['default']='default',
                 extract_fps=1, # Logic FPS for labeling
                 **kwargs):
        
        # We don't use the parent's complex processing anymore, just basic setup
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
        self.extract_fps = extract_fps 
        self.video_data = self.load_jsonl(expression_file)
        
        # Ensure processor is Qwen3VL type
        if not hasattr(self.preprocessor, 'video_processor'):
            raise ValueError("Preprocessor must be Qwen3VLProcessor")

    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def real_len(self):
        return len(self.video_data)

    def _get_video_frames_pil(self, video_path):
        """
        Reads video and returns list of PIL images + logical info.
        Qwen3VL processor handles the pixel formatting/resizing.
        """
        if not os.path.exists(video_path):
            return None, None, None
            
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            duration = total_frames / fps
            
            # Sampling strategy: Uniform sampling
            if total_frames >= self.sampled_frames:
                indices = np.linspace(0, total_frames - 1, self.sampled_frames, dtype=int)
            else:
                indices = np.arange(total_frames)
            
            indices = sorted(list(set(indices)))
            
            # Get Frames
            frames_arr = vr.get_batch(indices).asnumpy()
            images = [Image.fromarray(f) for f in frames_arr]
            
            # Calculate Logical Info for Labels
            frame_times = vr.get_frame_timestamp(indices)[:, 0]
            
            # Map physical frames to logical indices (for your VTG task labels)
            dense_indices = [int(t * self.extract_fps) for t in frame_times]
            dense_total_frames = int(duration * self.extract_fps)
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
        
        # 1. Get raw frames (PIL Images)
        frames, dense_indices, dense_total_frames = self._get_video_frames_pil(video_path)
        
        if frames is None:
            return self.prepare_data(random.randint(0, self.real_len() - 1))
            
        events = item['events']
        if len(events) == 0: return None
        
        # Select events
        if len(events) >= self.select_number:
            selected_indexes = np.random.choice(len(events), self.select_number, replace=False)
        else:
            selected_indexes = np.random.choice(len(events), self.select_number, replace=True)
        selected_events = [events[_idx] for _idx in selected_indexes]
            
        # 2. Construct Conversation Messages
        # We construct the list of messages first, then use processor
        messages = []
        
        # addition_prompt = f"The video has total {dense_total_frames} frames. The sampled indices is {dense_indices}."
        addition_prompt = ""
        
        # System prompt (optional, often handled inside apply_chat_template)
        # messages.append({"role": "system", "content": "You are a helpful assistant."})

        for i, event in enumerate(selected_events):
            query = event['query']
            span = event['span'][0] 
            
            # Calculate Labels
            start_frame_idx = int(span[0] * self.extract_fps)
            end_frame_idx = int(span[1] * self.extract_fps)
            start_frame_idx = min(max(start_frame_idx, 0), dense_total_frames - 1)
            end_frame_idx = min(max(end_frame_idx, 0), dense_total_frames - 1)
            if start_frame_idx > end_frame_idx: end_frame_idx = start_frame_idx
                
            answer_dict = {"start_frame": start_frame_idx, "end_frame": end_frame_idx}
            answer_str = json.dumps(answer_dict)
            
            question = random.choice(VTG_QUESTIONS).format(class_name=query)
            
            # For the first turn, include the video
            if i == 0:
                content = [
                    {
                        "type": "video",
                        "video": frames, # Pass PIL images list directly
                        "fps": 1.0, # Optional: Set fps for Qwen's internal time calculation if needed
                    },
                    {"type": "text", "text": f"{addition_prompt}\n{question}"}
                ]
            else:
                content = [{"type": "text", "text": query}]
                
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_str}]})

        # 3. Process inputs using Qwen3VL Logic
        try:
            # A. Prepare raw text with special tokens and vision placeholders
            text = self.preprocessor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            
            # B. Extract and Process Vision Inputs
            image_inputs, video_inputs = process_vision_info(messages)
            
            # C. Tokenize and Encapsulate
            # 'inputs' will contain input_ids, attention_mask, pixel_values_videos, video_grid_thw
            inputs = self.preprocessor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # 4. Generate Labels
            # We need to mask out the user prompt and system prompt in labels
            input_ids = inputs['input_ids'][0]
            labels = input_ids.clone()
            
            # Qwen uses <|im_start|>user ... <|im_end|><|im_start|>assistant ... <|im_end|>
            # We mask everything except the assistant's response.
            
            # Get special token ids
            im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            # Note: Qwen3 tokenizer might encode "assistant" as a separate token or part of string
            # It's safer to iterate through the sequence to find turns.
            
            # Simple State Machine for Masking
            # State 0: In User/System part -> Mask (Label = -100)
            # State 1: In Assistant part -> Keep Label
            
            # Assuming format: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
            
            # Find all indices of <|im_start|>
            im_start_indices = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
            
            current_idx = 0
            # By default mask everything
            labels[:] = -100
            
            for start_idx in im_start_indices:
                # Check the token immediately following <|im_start|> to see role
                # Note: This requires knowing how "user" and "assistant" are tokenized.
                # A robust way is to decode the next few tokens, but that's slow.
                # Qwen template usually: <|im_start|>assistant
                
                # Let's decode a small window to check role
                role_window = input_ids[start_idx+1 : start_idx+5]
                role_str = self.tokenizer.decode(role_window)
                
                if "assistant" in role_str:
                    # Find the next <|im_end|>
                    rest_ids = input_ids[start_idx:]
                    end_matches = (rest_ids == im_end_id).nonzero(as_tuple=True)[0]
                    if len(end_matches) > 0:
                        # The content starts after "<|im_start|>assistant\n"
                        # The exact offset depends on tokenizer.
                        # Let's heuristically say content starts where "assistant" ends.
                        # Actually, Qwen training usually masks the header "<|im_start|>assistant\n" too.
                        
                        # Find the first newline after start_idx? Or just unmask from start to end
                        # Standard SFT usually masks instruction but keeps response.
                        
                        # Let's search for the end of the header "\n"
                        # But for safety in Qwen, we often unmask from the token AFTER "assistant" header
                        # or just rely on finding the next <|im_end|>.
                        
                        abs_end_idx = start_idx + end_matches[0] + 1 # Include im_end in loss? usually yes
                        
                        # Determine where response starts (heuristic: skip a few tokens for header)
                        # A better way: The processor's chat template output has structure.
                        # Since we used apply_chat_template, we have the full string.
                        
                        # Simplified approach: Unmask from (start_idx + specific_offset) to abs_end_idx
                        # Offset for "<|im_start|>assistant\n" is roughly 2-3 tokens.
                        labels[start_idx+2 : abs_end_idx] = input_ids[start_idx+2 : abs_end_idx]
            
            # Remove the batch dimension for dataset output
            out_data_dict = {
                'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
                'labels': labels,
                'pixel_values_videos': inputs['pixel_values_videos'], # List or Tensor
                'video_grid_thw': inputs['video_grid_thw'], # Tensor
                'type': 'video'
            }
            
            return out_data_dict

        except Exception as e:
            print(f"Error processing Qwen3 native data at index {index}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None



if __name__ == '__main__':
    from transformers import AutoTokenizer, Qwen3VLProcessor
    from xtuner.utils import PROMPT_TEMPLATE
    from projects.sa2va.models import DirectResize
    
    dataset = Sa2VA07VTGDataset(
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