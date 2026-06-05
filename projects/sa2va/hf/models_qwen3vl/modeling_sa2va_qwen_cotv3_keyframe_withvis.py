import torch
from torch import nn
from transformers import Qwen3VLForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel
from .configuration_sa2va_chat import Sa2VAChatConfigQwen
from .sam2 import SAM2
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
import json
import re
import os
import time
from PIL import Image, ImageDraw, ImageFont
import cv2


def prepare_video_inputs(
    images,              # List[PIL.Image] or List[np.ndarray]
    sample_times,        # List[float], 对应每张图的时间戳
    text_prompt,         # str, 用户的问题/Prompt
    processor,           # Qwen3VLProcessor 实例
    tokenizer,           # AutoTokenizer 实例
    min_pixels=4 * 28 * 28,
    max_pixels=32 * 28 * 28,
    device="cuda"
):
    """
    手动构建 Qwen3-VL 的视频输入，确保与 Training 逻辑完全一致。
    """
    
    # --- 1. 图像预处理 (Resize using CV2) ---
    def _resize_image_to_limit(image, min_p, max_p):
        # 统一转为 numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        h, w = image.shape[:2]
        pixel_count = w * h
        
        if min_p <= pixel_count <= max_p:
            return image
            
        if pixel_count < min_p:
            target_pixels = min_p
        else:
            target_pixels = max_p
            
        ratio = (target_pixels / pixel_count) ** 0.5
        new_w = max(28, int(w * ratio))
        new_h = max(28, int(h * ratio))
        
        # 使用 cv2.INTER_CUBIC 保持与 dataset 一致
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    resized_frames = [_resize_image_to_limit(img, min_pixels, max_pixels) for img in images]

    # --- 2. 调用 Processor 获取特征 ---
    # do_sample_frames=False 禁用自动采样，确保特征与输入帧对应
    video_inputs = processor.video_processor(
        videos=resized_frames,
        return_metadata=True,
        do_sample_frames=False 
    )
    
    pixel_values_videos = video_inputs['pixel_values_videos'] # Tensor or List
    video_grid_thw = video_inputs['video_grid_thw'] # [[T, H, W]]
    
    # --- 3. 构建 Video Special String (Timestamp 注入) ---
    grid = video_grid_thw[0]
    grid_t, grid_h, grid_w = int(grid[0]), int(grid[1]), int(grid[2])
    
    # 时间戳重采样逻辑 (与 Dataset _sample_timestamps_to_grid 一致)
    if grid_t > 0:
        indices = np.linspace(0, len(sample_times) - 1, grid_t)
        merged_timestamps = [sample_times[int(i)] for i in indices]
    else:
        merged_timestamps = []

    video_token = "<|video_pad|>"
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    
    # 计算每个时间步的空间 token 数
    spatial_merge_size = processor.video_processor.merge_size
    spatial_tokens = (grid_h * grid_w) // (spatial_merge_size ** 2)
    
    video_special_str = ""
    for i in range(grid_t):
        t = merged_timestamps[i]
        video_special_str += f"<{t:.1f} seconds>{vision_start}{video_token * spatial_tokens}{vision_end}"

    # --- 4. 构建 Chat Template ---
    user_content = f"{video_special_str}\n{text_prompt}"
    user_part = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    full_text = user_part
    
    # --- 5. Tokenization ---
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 处理 Pixel Values 设备
    if isinstance(pixel_values_videos, torch.Tensor):
        pixel_values_videos = pixel_values_videos.to(device)
    else:
        pixel_values_videos = [pv.to(device) for pv in pixel_values_videos]

    return {
        "input_ids": input_ids_tensor,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": torch.tensor(video_grid_thw).to(device),
        "attention_mask": torch.ones_like(input_ids_tensor).to(device)
    }


class Sa2VAChatModelQwenCOT(PreTrainedModel):
    config_class = Sa2VAChatConfigQwen
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['Qwen3VisionTransformerPretrainedModel', 'Qwen3VLDecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: Sa2VAChatConfigQwen, model=None, use_flash_attn=True):
        super().__init__(config)
        
        self.SYS_PROMPT = "You are a helpful assistant. Your ultimate goal is to perform video temporal grounding (VTG) and referring video object segmentation (RefVOS). Based on the text query and video content, output your thinking process within <think> and </think> tags. If anything is unclear, you can select frames from the video for a clearer view by outputting <select>{\"start\": start, \"end\": end, \"keyframe\": keyframe}</select>, where 'start' and 'end' are the start and end frame indices of the region for detailed analysis, and 'keyframe' is the corresponding keyframe index. Once the final answer is confirmed, provide it within <answer><VTG>{\"start\": start, \"end\": end, \"keyframe\": keyframe}</VTG>, <RefSeg></RefSeg></answer>."
        
        self.min_pixel_temporal = 4*28*28 
        self.max_pixel_temporal = 32*28*28
        self.min_pixel_spatial = 4*28*28
        self.max_pixel_spatial = 256*28*28
        self.min_pixel_key = 4*28*28
        self.max_pixel_key = 512*28*28
        self.max_video_sample = 20
        self.max_select_K = 8
        self.torch_dtype = torch.bfloat16

        if model is not None:
            self.model = model
        else:
            self.model = Qwen3VLForConditionalGeneration(config)

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

    def _resize_image_to_limit(self, image, min_pixels, max_pixels):
        w, h = image.size
        pixel_count = w * h
        if min_pixels <= pixel_count <= max_pixels:
            return image
        if pixel_count < min_pixels:
            target_pixels = min_pixels
        else:
            target_pixels = max_pixels
        ratio = (target_pixels / pixel_count) ** 0.5
        new_w = max(28, int(w * ratio))
        new_h = max(28, int(h * ratio))
        return image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # --- 新增：保存图片并添加大字体标注 (拼接逻辑) ---
    def _save_images_with_annotations(self, images, indices, save_dir, filename_prefix, font_path="/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/arial.ttf"):
        """
        保存图片列表，并在图片上绘制大字体的帧号。如果是多张图，拼接成一张长图。
        """
        if not images: return None
        
        annotated_images = []
        
        # 动态计算统一高度 (以第一张为准，或者 max)
        target_h = max(img.height for img in images)
        
        # 准备字体
        # 字体高度占图片高度的 15%
        font_size = int(target_h * 0.15) 
        font_size = max(40, font_size) # 最小字号保障
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        for idx, img_pil in zip(indices, images):
            # Resize to same height if needed
            if img_pil.height != target_h:
                w, h = img_pil.size
                new_w = int(w * (target_h / h))
                img_pil = img_pil.resize((new_w, target_h), Image.Resampling.BILINEAR)
            
            # Draw Text
            draw = ImageDraw.Draw(img_pil)
            text = f"F:{idx}"
            
            # 描边逻辑：为了在任何背景上都清晰，画多次偏移
            stroke_width = max(2, font_size // 15)
            x, y = 20, 20
            
            # 绘制描边 (黑色)
            draw.text((x, y), text, font=font, fill=(0, 0, 0), stroke_width=stroke_width, stroke_fill=(0, 0, 0))
            # 绘制主体 (白色/亮色)
            draw.text((x, y), text, font=font, fill=(255, 255, 0)) # 黄色比较显眼

            annotated_images.append(img_pil)
        
        # 拼接图片
        total_w = sum(im.width for im in annotated_images)
        concat_img = Image.new('RGB', (total_w, target_h))
        
        curr_x = 0
        for im in annotated_images:
            concat_img.paste(im, (curr_x, 0))
            curr_x += im.width
            
        # 保存
        save_path = os.path.join(save_dir, f"{filename_prefix}_concat.jpg")
        concat_img.save(save_path)
        return f"{filename_prefix}_concat.jpg"

    def _save_visualization(self, specific_save_dir, turn_idx, text_content, image_files):
        if not specific_save_dir: return
        
        log_file = os.path.join(specific_save_dir, "conversation_log.md")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n## Turn {turn_idx} ({timestamp})\n")
            f.write("### Context / Interaction:\n")
            f.write(f"{str(text_content)}\n")
            
            if image_files:
                f.write("\n### Visual Inputs/Outputs:\n")
                for img_name in image_files:
                    f.write(f"![img]({img_name})\n") # Markdown 图片链接，直接用相对路径

    def _overlay_mask_on_image(self, image_pil, mask_np, alpha=0.5, color=(0, 255, 0)):
        if isinstance(image_pil, np.ndarray):
            image_pil = Image.fromarray(image_pil)
        image_pil = image_pil.convert("RGBA")
        
        if mask_np.dtype != np.uint8:
            mask_np = (mask_np * 255).astype(np.uint8)
        
        if len(mask_np.shape) == 3:
            mask_np = mask_np.squeeze(0)
            
        if mask_np.shape != image_pil.size[::-1]: 
             mask_image = Image.fromarray(mask_np, mode='L')
             mask_image = mask_image.resize(image_pil.size, resample=Image.NEAREST)
        else:
             mask_image = Image.fromarray(mask_np, mode='L')

        mask_layer = Image.new("RGBA", image_pil.size, color + (0,))
        mask_layer.putalpha(mask_image)
        
        blended = Image.alpha_composite(image_pil, mask_layer)
        final_image = Image.blend(image_pil, blended, alpha)
        return final_image.convert("RGB")
    
    def predict_forward(
            self,
            # Processed inputs
            processed_sam2_images=None,
            qwen_video_frames=None,
            qwen_spatial_frames=None,
            video_sample_index=None,
            image_sample_index=None,
            
            # Original inputs
            image=None,
            video=None, 
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
            max_turns=3,
            max_temporal_frames_per_round=10,
            
            # --- 可视化参数 ---
            vis_save_path=None,
            sample_id=None,
            lang_embed_injecting_per_frame=False
    ):
        assert processor is not None
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.seg_token_idx = self.tokenizer.convert_tokens_to_ids('[SEG]')

        clean_text = text.replace('<image>', "").replace('<video>', "")

        if video_sample_index is None or image_sample_index is None:
             video_sample_index = np.arange(len(video)) if len(video) < self.max_video_sample else np.linspace(0, len(video)-1, self.max_video_sample, dtype=int)
             image_sample_index = np.arange(len(video)) if len(video) < self.max_select_K else np.linspace(0, len(video)-1, self.max_select_K, dtype=int)

        # --- 可视化：初始化 ---
        current_sample_vis_dir = None
        if vis_save_path:
            if sample_id is not None:
                current_sample_vis_dir = os.path.join(vis_save_path, str(sample_id))
            else:
                current_sample_vis_dir = os.path.join(vis_save_path, f"sample_{str(time.time()).replace('.', '')}")
            
            os.makedirs(current_sample_vis_dir, exist_ok=True)
            
            # 保存 Turn 0 (Initial) 的拼接图片
            vis_files = []
            if qwen_video_frames:
                f_name = self._save_images_with_annotations(
                    qwen_video_frames, video_sample_index, current_sample_vis_dir, "turn_0_temporal"
                )
                if f_name: vis_files.append(f_name)
            
            if qwen_spatial_frames:
                f_name = self._save_images_with_annotations(
                    qwen_spatial_frames, image_sample_index, current_sample_vis_dir, "turn_0_keyframe"
                )
                if f_name: vis_files.append(f_name)
            
            self._save_visualization(current_sample_vis_dir, 0, f"User Query: {clean_text}", vis_files)

        input_dict = {}
        ret_masks = []
        masks_shot = []
        
        # --- 1. SAM2 Feature Processing ---
        if processed_sam2_images is not None:
            extra_pixel_values = []
            for g_image_np in processed_sam2_images:
                g_image = torch.from_numpy(g_image_np).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_image)
            
            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
            ]).to(self.torch_dtype).to(self.device)
            input_dict['g_pixel_values'] = g_pixel_values

        # --- 2. Qwen Prompt Construction ---
        video_content = [{"type": "image", "image": img} for img in (qwen_video_frames if qwen_video_frames else [])]
        image_content = [{"type": "image", "image": img} for img in (qwen_spatial_frames if qwen_spatial_frames else [])]
        
        video_text = f"\nHere are {len(video_sample_index)} low-resolution frames in this video (frame indice is {str(list(video_sample_index))})\n"
        image_text = f"Here are {len(image_sample_index)} high-resolution frames in this video (frame indice:{str(list(image_sample_index))}).\n"
        
        if not video_content: video_text = ""
        if not image_content: image_text = ""

        content = (
            video_content + 
            [{"type": "text", "text": video_text}] + 
            image_content + 
            [{"type": "text", "text": image_text}, {"type": "text", "text": clean_text}]
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.SYS_PROMPT}]},
            {"role": "user", "content": content},
        ]

        # --- CoT Loop ---
        current_turn = 0
        final_predict = ""
        keyframe_idx = None

        while current_turn < max_turns:
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            mm_inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            mm_inputs = mm_inputs.to(self.device)

            generate_output = self.model.generate(
                **mm_inputs,
                max_new_tokens=1024,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
            )

            generate_output_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
            ]
            predict = self.processor.batch_decode(generate_output_trimmed, skip_special_tokens=False)[0].strip()
            final_predict = predict 
            
            messages.append({"role": "assistant", "content": predict.replace('<|im_end|>', '')})

            # --- Tool Use Parsing ---
            select_match = re.search(r"<select>(.*?)</select>", predict, re.DOTALL)

            if select_match:
                try:
                    select_match_vtg = re.search(r"<VTG>(.*?)</VTG>", select_match.group(1), re.DOTALL)
                    select_str = select_match_vtg.group(1) if select_match_vtg else select_match.group(1)
                    
                    select_data = json.loads(select_str)
                    start_idx = int(select_data.get("start"))
                    end_idx = int(select_data.get("end"))
                    keyframe_idx = int(select_data.get("keyframe"))
                    
                    start_idx = max(0, start_idx)
                    end_idx = min(len(video) - 1, end_idx)
                    keyframe_idx = max(0, min(len(video) - 1, keyframe_idx))

                    if end_idx < start_idx: 
                        break 
                    
                    tool_content = []
                    
                    # 准备新的图片数据
                    new_temporal_imgs = []
                    new_temporal_indices = []
                    new_keyframe_img = None
                    
                    # Strategy 1: Temporal
                    if (end_idx - start_idx + 1) <= max_temporal_frames_per_round:
                        temporal_indices = np.arange(start_idx, end_idx + 1, dtype=int)
                    else:
                        temporal_indices = np.linspace(start_idx, end_idx, max_temporal_frames_per_round, dtype=int)
                    
                    for t_idx in temporal_indices:
                        resized_img = self._resize_image_to_limit(
                            video[t_idx], self.min_pixel_temporal, self.max_pixel_temporal
                        )
                        tool_content.append({"type": "image", "image": resized_img})
                        new_temporal_imgs.append(resized_img)
                        new_temporal_indices.append(t_idx)
                    
                    temporal_text = f"\nHere are {len(temporal_indices)} low-resolution frames in this video (frame indices are {str(list(temporal_indices))})\n"
                    tool_content.append({"type": "text", "text": temporal_text})

                    # Strategy 2: Keyframe
                    kf_img = self._resize_image_to_limit(
                        video[keyframe_idx], self.min_pixel_key, self.max_pixel_key
                    )
                    tool_content.append({"type": "image", "image": kf_img})
                    new_keyframe_img = kf_img
                    
                    kf_text = f"Here is key frames (frame indice:{keyframe_idx}).\n"
                    tool_content.append({"type": "text", "text": kf_text})

                    # Strategy 3: Instruction
                    instruction_text = (
                        f"The text query is '{clean_text}'.\n "
                        "Continue your reasoning process inside <think> and </think>. "
                        "If needed, you can continue to select images on the original video, by outputting <select> and </select> as before. "
                        "If the final answer is confirmed, put your final answer inside <answer> and </answer>."
                    )
                    tool_content.append({"type": "text", "text": instruction_text})

                    messages.append({"role": "user", "content": tool_content})
                    
                    # --- 可视化 Tool Outputs ---
                    if current_sample_vis_dir:
                        vis_files = []
                        # 拼接 temporal
                        f_temp = self._save_images_with_annotations(
                            new_temporal_imgs, new_temporal_indices, current_sample_vis_dir, f"turn_{current_turn + 1}_temporal"
                        )
                        if f_temp: vis_files.append(f_temp)
                        
                        # 保存 keyframe
                        f_kf = self._save_images_with_annotations(
                            [new_keyframe_img], [keyframe_idx], current_sample_vis_dir, f"turn_{current_turn + 1}_keyframe"
                        )
                        if f_kf: vis_files.append(f_kf)
                        
                        # 记录这一轮的对话和新图片
                        self._save_visualization(
                            current_sample_vis_dir, 
                            current_turn + 1, 
                            f"Assistant Output: {predict}\n\nTool Call: Temporal {list(temporal_indices)}, Keyframe {keyframe_idx}", 
                            vis_files
                        )

                    current_turn += 1
                    continue

                except Exception as e:
                    print(f"Error parsing JSON data in select block: {e}")
                    break
            else:
                # No select tool used, loop ends
                break

        # --- Final Inference ---
        if image is None and video is None and '<image>' not in past_text:
             return {'prediction': final_predict, 'prediction_masks': ret_masks}
        
        if keyframe_idx is None:
            try:
                select_str = re.search(r"<VTG>(.*?)</VTG>", final_predict, re.DOTALL)
                if select_str:
                    select_data = json.loads(select_str.group(1))
                    final_keyframe_idx = int(select_data.get("keyframe"))
                    final_keyframe_idx = max(0, min(final_keyframe_idx, len(video) - 1))
                else:
                    final_keyframe_idx = image_sample_index[len(image_sample_index) // 2]
            except:
                final_keyframe_idx = image_sample_index[len(image_sample_index) // 2]
        else:
            final_keyframe_idx = keyframe_idx

        # SAM2 Decoding
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states[-1:])

        ori_image_size = video[0].size if video else (1024, 1024)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            
            pred_masks, pred_masks_shot = self.grounding_encoder.language_embd_inference_keyframe_bidirection(
                sam_states, 
                [seg_hidden_states], 
                final_keyframe_idx,
                lang_embed_injecting_per_frame=lang_embed_injecting_per_frame
            )

            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks_shot = F.interpolate(pred_masks_shot, size=(h, w), mode='bilinear', align_corners=False)
            
            masks = masks[:, 0].sigmoid() > 0.5
            masks_shot = masks_shot[:, 0].sigmoid() > 0.5
            
            # --- 可视化：保存最终结果 ---
            if current_sample_vis_dir:
                mask_np = masks_shot.cpu().numpy()[0]
                if video and final_keyframe_idx < len(video):
                    ori_img = video[final_keyframe_idx]
                    overlay_img = self._overlay_mask_on_image(ori_img, mask_np, color=(255, 0, 0))
                    
                    # 使用大字体标注 Final 帧号
                    f_final = self._save_images_with_annotations(
                        [overlay_img], [final_keyframe_idx], current_sample_vis_dir, f"final_result_keyframe_{final_keyframe_idx}"
                    )
                    
                    self._save_visualization(
                        current_sample_vis_dir, 
                        "Final", 
                        f"Final Prediction: {final_predict}\nKeyframe Index: {final_keyframe_idx}", 
                        [f_final]
                    )

            ret_masks.append(masks.cpu().numpy())
            masks_shot = masks_shot.cpu().numpy()

        return {'prediction': messages, 'prediction_masks': ret_masks, 'prediction_masks_shot': masks_shot, "sample_index": image_sample_index, "keyframe_index": [final_keyframe_idx]}

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