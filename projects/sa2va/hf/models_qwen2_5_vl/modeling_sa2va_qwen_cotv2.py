import torch
from torch import nn
from transformers import (AutoModel, GenerationConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen2ForCausalLM)
from transformers.modeling_utils import PreTrainedModel

from .configuration_sa2va_chat import Sa2VAChatConfigQwen

from .sam2 import SAM2

import numpy as np
from torchvision.transforms.functional import to_pil_image

import torch.nn.functional as F

from qwen_vl_utils import process_vision_info

from PIL import Image
import re
import json



class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModelQwenCOTV2(PreTrainedModel):
    config_class = Sa2VAChatConfigQwen
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['Qwen2_5_VisionTransformerPretrainedModel', 'Qwen2_5_VLDecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True



    def __init__(self, config: Sa2VAChatConfigQwen, model=None, use_flash_attn=True):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1024, )

        self.SYS_PROMPT = "You are a helpful assistant. Your ultimate goal is to perform video temporal grounding (VTG) and referring video object segmentation (RefVOS). Based on the text query and video content, output your thinking process within <think> and </think> tags. If anything is unclear, you can select frames from the video for a clearer view by outputting <select>{\"start\": start, \"end\": end, \"keyframe\": keyframe}</select>, where 'start' and 'end' are the start and end frame indices of the region for detailed analysis, and 'keyframe' is the corresponding keyframe index. Once the final answer is confirmed, provide it within <answer><VTG>{\"start\": start, \"end\": end, \"keyframe\": keyframe}</VTG>, <RefSeg></RefSeg></answer>."

        self.min_pixel_temporal = 8*28*28 
        self.max_pixel_temporal = 16*28*28

        self.min_pixel_spatial = 128*28*28
        self.max_pixel_spatial = 256*28*28

        self.max_video_sample = 0
        self.max_select_K = 5

        self.torch_dtype = torch.bfloat16

        if model is not None:
            self.model=model
        else:
            self.model = Qwen2_5_VLForConditionalGeneration(config)

        llm_hidden_size = config.text_config.hidden_size

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    # 辅助函数：按像素限制等比缩放
    def _resize_image_to_limit(self, image, min_pixels, max_pixels):
        w, h = image.size
        pixel_count = w * h
        
        # 如果在范围内，不处理 (或者你可以强制 resize 到更小的尺寸以节省 token)
        if min_pixels <= pixel_count <= max_pixels:
            return image
            
        # 计算缩放比例
        if pixel_count < min_pixels:
            #太小了，放大 (通常对于 temporal 不需要放大，这里主要是防止太小报错，或者你可以直接返回原图)
            target_pixels = min_pixels
        else:
            # 太大了，缩小
            target_pixels = max_pixels
            
        ratio = (target_pixels / pixel_count) ** 0.5
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # 确保长宽至少为 28 (Qwen patch size)
        new_w = max(28, new_w)
        new_h = max(28, new_h)
        
        return image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
    ):
        assert processor is not None
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.seg_token_idx = self.tokenizer.convert_tokens_to_ids('[SEG]')

        text = text.replace('<image>', "").replace('<video>', "")

        # --- 1. 初始采样 ---
        if len(video) > self.max_video_sample:
            video_sample_index = np.linspace(0, len(video)-1, self.max_video_sample, dtype=int)
        else:
            video_sample_index = np.arange(len(video))
        
        if len(video) > self.max_video_sample:
            image_sample_index = np.linspace(0, len(video)-1, self.max_select_K, dtype=int)
        else:
            image_sample_index = np.arange(len(video))

        input_dict = {}
        ret_masks = []
        extra_pixel_values = []
        
        # 关键修改：直接准备 resize 好的 PIL Image 对象列表
        video_content_resized = [] 
        image_content_resized = []

        if video is not None:
            ori_image_size = video[0].size
            for frame_idx, frame_image in enumerate(video):
                # SAM2 逻辑不变
                g_image = np.array(frame_image) 
                g_image = self.extra_image_processor.apply_image(g_image)
                g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_image)

                # --- 核心修改：手动 Resize ---
                if frame_idx in video_sample_index:
                    # 缩放 Temporal 帧 (低分)
                    resized_img = self._resize_image_to_limit(
                        frame_image, self.min_pixel_temporal, self.max_pixel_temporal
                    )
                    video_content_resized.append(resized_img)
                
                if frame_idx in image_sample_index:
                    # 缩放 Spatial 帧 (高分)
                    resized_img = self._resize_image_to_limit(
                        frame_image, self.min_pixel_spatial, self.max_pixel_spatial
                    )
                    image_content_resized.append(resized_img)
            
            # 构建 messages
            video_content = [{"type": "image", "image": img} for img in video_content_resized]
            image_content = [{"type": "image", "image": img} for img in image_content_resized]
            
            video_text = "\nHere are {} low-resolution frames sampled at equal intervals from a video\n".format(len(video_sample_index))
            image_text = "Here are {} high-resolution frames in this video (frame indice:{})\n.".format(len(image_sample_index), image_sample_index)
            
            content = (
                [{"type": "text", "text": self.SYS_PROMPT}] +
                video_content + 
                [{"type": "text", "text": video_text}] + 
                image_content + 
                [{"type": "text", "text": image_text}, {"type": "text", "text": text}]
            )

            # SAM2 Inputs
            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
            ]).to(self.torch_dtype)
            input_dict['g_pixel_values'] = g_pixel_values

        messages = [
            {"role": "user", "content":content},
        ]

        # CoT Loop
        max_turns = 5
        current_turn = 0
        final_predict = ""
        
        while current_turn < max_turns:
            # 处理 Chat Template
            # print(messages)
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            # --- 核心修改：让 Processor 自由处理 ---
            mm_inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                min_pixels=self.min_pixel_temporal,
                max_pixels=self.max_pixel_spatial 
            )
            mm_inputs = mm_inputs.to(self.device)

            # Generate
            generate_output = self.model.generate(
                **mm_inputs,
                max_new_tokens=2048,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

            generate_output_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
            ]
            predict = self.processor.batch_decode(generate_output_trimmed, skip_special_tokens=False)[0].strip()
            final_predict = predict 
            
            messages.append({"role": "assistant", "content": predict.replace('<|im_end|>', '')}) # TODO 是否需要把结束符号<|im_end|>移除

            # 解析 Tool Use
            select_match = re.search(r"<select>(.*?)</select>", predict, re.DOTALL)
            
            if select_match:
                try:
                    select_match_vtg = re.search(r"<VTG>(.*?)</VTG>", select_match.group(1), re.DOTALL)
                    select_data = json.loads(select_match_vtg.group(1))
                    start_idx = int(select_data.get("start", 0))
                    end_idx = int(select_data.get("end", 0))
                    # keyframe_idx = int(select_data.get("keyframe", 0))
                    
                    tool_content = []
                    
                    # Tool frames
                    span_indices = np.linspace(start_idx, end_idx, num=min(3, end_idx - start_idx + 1), dtype=int)
                    tool_content.append({"type": "text", "text": "\n Here are the {} frames in this video\n".format(span_indices)})
                    
                    for t_idx in span_indices:
                        t_idx = min(max(0, t_idx), len(video) - 1)
                        resized_img = self._resize_image_to_limit(
                            video[t_idx], self.min_pixel_spatial, self.max_pixel_spatial
                        )
                        tool_content.append({"type": "image", "image": resized_img})
                    tool_content.append({"type": "text", "text": "\n Here are the {} frames in this video.\n Continue your reasoning process inside <think> and </think>. If needed, you can continue to select images on the original video, by outputting <select> and </select> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>.".format(span_indices)})

                    messages.append({"role": "user", "content": tool_content})
                    current_turn += 1
                    continue
                except Exception as e:
                    print(f"Error parsing JSON data in select block: {e}")
                    print(select_match_vtg)
            else:
                break
        
        # print(messages)
        select_VTG_match = re.search(r"<VTG>(.*?)</VTG>", predict, re.DOTALL)
        try:
            select_data = json.loads(select_VTG_match.group(1))
            start_idx = int(select_data.get("start", 0))
            end_idx = int(select_data.get("end", 0))
            keyframe_idx = int(select_data.get("keyframe", 0))
        except Exception as e:
            print(f"Error parsing VTG results: {e}")
            keyframe_idx = len(video)//2
            print(predict)

        # --- Final Segmentation Inference ---
        if image is None and video is None and '<image>' not in past_text:
             return {'prediction': final_predict, 'prediction_masks': ret_masks}

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states[-1:])

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * len(image_sample_index), image_sample_index)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': messages, 'prediction_masks': ret_masks}


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    # Ensure lengths match before indexing
    if len(hidden_states) != len(output_ids):
        # Fallback or alignment adjustment usually needed here if cache logic differs
        # Assuming non-cached standard generation where hidden_states map 1:1 to generated tokens
        min_len = min(len(hidden_states), len(output_ids))
        hidden_states = hidden_states[-min_len:]
        output_ids = output_ids[-min_len:]
        seg_mask = output_ids == seg_id
        
    return hidden_states[seg_mask]

    