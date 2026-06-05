import torch
from torch import nn
from transformers import (AutoModel, GenerationConfig, Qwen3VLForConditionalGeneration,
                          Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration)
from transformers.modeling_utils import PreTrainedModel

from .configuration_sa2va_chat import Sa2VAChatConfigQwen
from typing import List, Optional, Tuple, Union
from .sam2 import SAM2

import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor

import torch.nn.functional as F

from qwen_vl_utils import process_vision_info
import json
import re
from PIL import Image
import copy, gc

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModelQwenCOTV2(PreTrainedModel):
    config_class = Sa2VAChatConfigQwen
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['Qwen3VisionTransformerPretrainedModel', 'Qwen3VLDecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: Sa2VAChatConfigQwen, model=None, use_flash_attn=True):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1024, )

        self.SYS_PROMPT = "You are a helpful assistant. Your ultimate goal is to perform video temporal grounding (VTG) and referring video object segmentation (RefVOS). Based on the text query and video content, output your thinking process within <think> and </think> tags. If anything is unclear, you can select frames from the video for a clearer view by outputting <select>{\"start\": start, \"end\": end, \"keyframe\": keyframe}</select>, where 'start' and 'end' are the start and end frame indices of the region for detailed analysis, and 'keyframe' is the corresponding keyframe index. Once the final answer is confirmed, provide it within <answer><VTG>{\"start\": start, \"end\": end, \"keyframe\": keyframe}</VTG>, <RefSeg></RefSeg></answer>."

        self.min_pixel_temporal = 4*28*28 
        self.max_pixel_temporal = 32*28*28

        self.min_pixel_spatial = 4*28*28
        self.max_pixel_spatial = 128*28*28

        self.min_pixel_key = 4*28*28
        self.max_pixel_key = 256*28*28

        self.max_video_sample = 10
        self.max_select_K = 5

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

        self.max_turn = 3

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

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
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        new_w = max(28, new_w)
        new_h = max(28, new_h)
        return image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    def _decode_masks_for_turn(self, hidden_states, token_ids, sam_feats, video_len, keyframe_feats=None):
        """
        Decode both video masks and keyframe masks when keyframe features are available.
        """
        seg_token_id = self.processor.tokenizer.convert_tokens_to_ids('[SEG]')
        
        seq_len_h = hidden_states.shape[1]
        seq_len_t = token_ids.shape[1]
        min_len = min(seq_len_h, seq_len_t)
        
        curr_hidden = hidden_states[:, -min_len:, :]
        curr_ids = token_ids[:, -min_len:]

        seg_indices = (curr_ids[0] == seg_token_id).nonzero(as_tuple=True)[0]
        num_objs = len(seg_indices)

        if num_objs == 0:
            return None, None

        projected_states = self.text_hidden_fcs(curr_hidden)
        obj_emb = projected_states[0, seg_indices] # [N_obj, C]

        if sam_feats is None:
            return None, None

        base_sam_states = self.grounding_encoder.get_sam2_embeddings_expand(sam_feats, expand_size=num_objs)
        
        frames_per_batch = [video_len]
        pred_embeddings_video = self.generate_video_pred_embeddings([obj_emb], frames_per_batch)
        language_embeddings = torch.cat(pred_embeddings_video, dim=0)[:, None] # [T*N_obj, 1, C]

        pred_masks_flat = self.grounding_encoder.inject_language_embd_nf_nobj(
            base_sam_states,
            language_embeddings,
            nf_nobj=(video_len, num_objs)
        )
        
        H, W = pred_masks_flat.shape[-2:]
        video_masks = pred_masks_flat.view(video_len, num_objs, H, W)
        
        keyframe_mask = None
        if keyframe_feats is not None:
            # keyframe_feats shape: [1, 1, 256, 64, 64] (Batch=1, Time=1)
            # Expand states for N objects
            kf_base_states = self.grounding_encoder.get_sam2_embeddings_expand(
                keyframe_feats, expand_size=num_objs
            )
            
            # Prompt: [N_obj, C] -> [N_obj, 1, C] (Time=1)
            kf_lang_emb_list = self.generate_video_pred_embeddings([obj_emb], [1])
            kf_lang_emb = torch.cat(kf_lang_emb_list, dim=0)[:, None]
            
            # Decode
            kf_pred_masks_flat = self.grounding_encoder.inject_language_embd_nf_nobj(
                kf_base_states,
                kf_lang_emb,
                nf_nobj=(1, num_objs)
            )
            
            # Reshape: [1*N_obj, 1, H, W] -> [N_obj, H, W]
            H_kf, W_kf = kf_pred_masks_flat.shape[-2:]
            keyframe_mask = kf_pred_masks_flat.view(num_objs, H_kf, W_kf)

        return video_masks, keyframe_mask

    def _load_and_process_keyframe_tensor(self, image_path):
        """
        Load an image, resize it to 1024x1024, and convert it to the tensor format expected by SAM2.
        Used for:
        1. Inference: extract features for the SAM2 decoder.
        2. Training: keep tensors for forward_train loss computation.
        """
        try:
            # 1. Load & RGB
            pil_img = Image.open(image_path).convert('RGB')
            
            # 2. Resize (Using existing DirectResize logic or custom logic)
            img_np = np.array(pil_img)
            if hasattr(self, 'extra_image_processor'):
                img_np = self.extra_image_processor.apply_image(img_np)
            else:
                # Fallback resize if extra_image_processor is missing
                img_np = np.array(pil_img.resize((1024, 1024)))

            # 3. To Tensor [C, H, W]
            tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
            
            # 4. Preprocess (Normalize) -> [1, C, H, W] 
            # preprocess_image returns [C, H, W] normalized usually, we stack later or unsqueeze here if needed
            tensor = self.grounding_encoder.preprocess_image(tensor)
            
            return tensor.to(self.torch_dtype).to(self.device)
        except Exception as e:
            print(f"Error loading keyframe {image_path}: {e}")
            return torch.zeros((3, 1024, 1024), dtype=self.torch_dtype, device=self.device)

    def generation_forward_cot(self, processor, raw_inputs_list, sam_images_tensor, **kwargs):
        self.processor = processor 
        max_temporal_frames_per_round = kwargs.get('max_temporal_frames_per_round', 5)

        batch_size = len(raw_inputs_list)
        device = self.device

        def _clean_memory():
            gc.collect()
            torch.cuda.empty_cache()

        # 1. Init Containers
        batch_messages = []
        batch_images = [] 
        batch_total_frame_paths = []
        
        # Output Containers
        out_masks = [[] for _ in range(batch_size)]
        out_keyframe_masks = [[] for _ in range(batch_size)] 
        out_metadata = [[] for _ in range(batch_size)]
        
        # [New Containers]
        out_keyframe_tensors = [[] for _ in range(batch_size)]
        out_keyframe_indices = [[] for _ in range(batch_size)]
        
        final_states = [None] * batch_size
        out_num_objects = [[] for _ in range(batch_size)]
        
        batch_pending_keyframe_feats = [None] * batch_size
        batch_pending_global_k_indices = [None] * batch_size 
        
        active_indices = list(range(batch_size))
        
        for b, item in enumerate(raw_inputs_list):
            msgs = json.loads(item['prompt']) if isinstance(item['prompt'], str) else copy.deepcopy(item['prompt'])
            imgs = item.get('image', []) if isinstance(item.get('image', []), list) else [item.get('image')]
            tmp_imgs = item.get('temporal_image', []) if isinstance(item.get('temporal_image', []), list) else [item.get('temporal_image')]
            total_frames = item.get('total_frame_path', [])
            temporal_image_index = item.get('temporal_image_index', [])
            # image_index = item.get('image_index', [])
            
            batch_messages.append(msgs)
            batch_images.append(tmp_imgs+imgs)
            batch_total_frame_paths.append(total_frames)
            
            # Extract User Query (Log purpose)
            init_user_text = ""
            for m in reversed(msgs):
                if m['role'] == 'user':
                    if isinstance(m['content'], str):
                        init_user_text = m['content']
                    elif isinstance(m['content'], list):
                        init_user_text = "\n".join([c['text'] for c in m['content'] if c['type'] == 'text'])
                    break
            init_user_text = init_user_text.replace('<image>', "").replace('<video>', "")
            
            out_metadata[b].append({
                "turn_idx": 0,
                "user_query": init_user_text[:50] + "...",
                "frame_info": "Initial Sampling",
                "temporal_indices": str(list(temporal_image_index)),
            })
            
            final_states[b] = {"messages": copy.deepcopy(msgs), "images": copy.deepcopy(imgs), "tmporal_images": copy.deepcopy(tmp_imgs)}

            # ================= [Turn 0: Default Keyframe Initialization] =================
            init_k_idx = 0
            if total_frames and len(total_frames) > 0:
                init_k_idx = len(total_frames) // 2
                
                batch_pending_global_k_indices[b] = init_k_idx
                
                kf_path = total_frames[init_k_idx]
                kf_tensor = self._load_and_process_keyframe_tensor(kf_path) # [C, H, W]
                
                out_keyframe_tensors[b].append(kf_tensor)
                
                with torch.no_grad():
                    # Input to SAM2 requires batch dim: [1, C, H, W]
                    kf_feat = self.grounding_encoder.get_sam2_forward_images(kf_tensor.unsqueeze(0))
                    batch_pending_keyframe_feats[b] = kf_feat
            # =============================================================================

        # 2. Pre-compute Video Features
        batch_sam_feats = [None] * batch_size
        batch_vid_lens = [0] * batch_size
        
        if sam_images_tensor is not None:
            # sam_images_tensor shape: [B, T, 3, H, W]
            video_t_first = sam_images_tensor[0] 
            vid_len = len(video_t_first)

            with torch.no_grad():
                first_sample_feats = self.grounding_encoder.get_sam2_forward_images(video_t_first)

            for g_idx in range(batch_size):
                batch_vid_lens[g_idx] = vid_len
                batch_sam_feats[g_idx] = first_sample_feats

            del video_t_first

        # 3. Rollout Loop
        max_turns = self.max_turn
        gen_args = {k:v for k,v in kwargs.items() if k not in ['input_ids', 'images', 'pixel_values', 'attention_mask', 'image_grid_thw', 'video_grid_thw', 'past_key_values']}

        for turn in range(max_turns):
            if not active_indices:
                break
            
            # --- A. Prepare Inputs ---
            current_batch_texts = []
            current_batch_images = []
            
            for b in active_indices:
                text = processor.apply_chat_template(batch_messages[b], tokenize=False, add_generation_prompt=True)
                current_batch_texts.append(text)
                current_batch_images.append(batch_images[b] if batch_images[b] else None)

            inputs = processor(
                text=current_batch_texts,
                images=current_batch_images,
                return_tensors="pt",
                padding=True
            ).to(device)
            input_len = inputs.input_ids.shape[1]

            # --- B. LLM Generation ---
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    **gen_args
                )

            generated_ids = out.sequences[:, input_len:]
            last_layer_hidden_steps = [step[-1] for step in out.hidden_states]
            batched_hidden_states = torch.cat(last_layer_hidden_steps, dim=1)

            del inputs

            next_active_indices = []

            for local_idx, global_idx in enumerate(active_indices):
                gen_seq = generated_ids[local_idx]
                hid_state = batched_hidden_states[local_idx:local_idx+1]
                gen_seq_expanded = gen_seq.unsqueeze(0)
                
                # --- C. Mask Decoding ---
                sam_feat = batch_sam_feats[global_idx]
                vid_len = batch_vid_lens[global_idx]
                
                pending_kf_feat = batch_pending_keyframe_feats[global_idx]
                
                mask_result, kf_mask_result = self._decode_masks_for_turn(
                    hid_state, gen_seq_expanded[:, :-1], sam_feat, vid_len, 
                    keyframe_feats=pending_kf_feat 
                )

                current_turn_obj_count = 0
                if mask_result is not None:
                    # mask_result: [Video_Len, Num_Objs, H, W]
                    num_objs = mask_result.shape[1]
                    current_turn_obj_count = num_objs
                    
                    for obj_idx in range(num_objs):
                        out_masks[global_idx].append(mask_result[:, obj_idx, :, :])
                
                out_num_objects[global_idx].append(current_turn_obj_count)
                
                
                current_global_k_idx = -1
                if batch_pending_global_k_indices[global_idx] is not None:
                    current_global_k_idx = batch_pending_global_k_indices[global_idx]
                
                out_keyframe_indices[global_idx].append(current_global_k_idx)
                
                out_keyframe_masks[global_idx].append(kf_mask_result)
                
                batch_pending_keyframe_feats[global_idx] = None
                batch_pending_global_k_indices[global_idx] = None
                
                # --- D. Response Parsing ---
                resp = processor.decode(gen_seq, skip_special_tokens=False).replace('<|im_end|>', '').replace('<|endoftext|>', '')
                
                batch_messages[global_idx].append({"role": "assistant", "content": resp})
                final_states[global_idx]["messages"].append({"role": "assistant", "content": resp})

                # --- E. Select Logic & Feature Prep for NEXT TURN ---
                sel = re.search(r"<select>.*?<VTG>(.*?)</VTG>.*?</select>", resp, re.DOTALL)
                should_continue = False
                
                if sel and turn < max_turns - 1:
                    total_frames = batch_total_frame_paths[global_idx]
                    
                    if total_frames:
                        try:
                            d = json.loads(sel.group(1))
                            start_idx = int(d.get('start', 0))
                            end_idx = int(d.get('end', 0))
                            keyframe_idx = int(d.get('keyframe', 0))
                            
                            max_idx = len(total_frames) - 1
                            start_idx = max(0, start_idx)
                            end_idx = min(max_idx, end_idx)
                            keyframe_idx = max(0, min(max_idx, keyframe_idx))

                            if end_idx >= start_idx:
                                tool_content = []
                                new_pil_images = []

                                # 1. Temporal Sampling (Low Res for LLM)
                                if (end_idx - start_idx + 1) <= max_temporal_frames_per_round:
                                    temporal_indices = np.arange(start_idx, end_idx + 1, dtype=int)
                                else:
                                    temporal_indices = np.linspace(start_idx, end_idx, max_temporal_frames_per_round, dtype=int)
                                temporal_indices = temporal_indices.tolist()
                                for t_idx in temporal_indices:
                                    img_path = total_frames[t_idx]
                                    img = Image.open(img_path).convert('RGB')
                                    resized_img = self._resize_image_to_limit(
                                        img, self.min_pixel_temporal, self.max_pixel_temporal
                                    )
                                    new_pil_images.append(resized_img)
                                    tool_content.append({"type": "image"})
                                
                                tool_content.append({"type": "text", "text": f"\nHere are {len(temporal_indices)} low-resolution frames in this video (frame indices are {str(list(temporal_indices))})\n"})

                                # 2. Keyframe Selection (High Res for LLM & SAM2)
                                kf_path = total_frames[keyframe_idx]
                                kf_img_raw = Image.open(kf_path).convert('RGB')
                                
                                # 2.1 Low Res for LLM Context
                                kf_img_resized_pil = self._resize_image_to_limit(
                                    kf_img_raw, self.min_pixel_key, self.max_pixel_key
                                )
                                new_pil_images.append(kf_img_resized_pil)
                                tool_content.append({"type": "image"})
                                tool_content.append({"type": "text", "text": f"Here is key frames (frame indice:{keyframe_idx}).\n"})

                                # ================= [Next Turn Preparation] =================
                                batch_pending_global_k_indices[global_idx] = keyframe_idx
                                
                                kf_tensor = self._load_and_process_keyframe_tensor(kf_path)
                                out_keyframe_tensors[global_idx].append(kf_tensor)
                                
                                with torch.no_grad():
                                    kf_feat = self.grounding_encoder.get_sam2_forward_images(kf_tensor.unsqueeze(0))
                                    batch_pending_keyframe_feats[global_idx] = kf_feat
                                # ===========================================================

                                query_text = raw_inputs_list[global_idx]["problem"]
                                instruction_text = (
                                    f"The text query is '{query_text}'.\n "
                                    "Continue your reasoning process inside <think> and </think>. "
                                    "If needed, you can continue to select images on the original video, by outputting <select> and </select> as before. "
                                    "If the final answer is confirmed, put your final answer inside <answer> and </answer>."
                                )
                                tool_content.append({"type": "text", "text": instruction_text})

                                out_metadata[global_idx].append({
                                    "turn_idx": turn + 1,
                                    "user_query": f"Sampled {len(temporal_indices)} temporal + 1 keyframe",
                                    "temporal_indices": str(list(temporal_indices)),
                                    "keyframe_index": keyframe_idx
                                })
                                
                                new_msg = {"role": "user", "content": tool_content}
                                
                                batch_messages[global_idx].append(new_msg)
                                batch_images[global_idx].extend(new_pil_images)
                                final_states[global_idx]["messages"].append(new_msg)
                                final_states[global_idx]["images"].extend(new_pil_images)
                                should_continue = True

                        except Exception as e:
                            print(f"Select parse error in batch {global_idx}: {e}")
                
                if should_continue:
                    next_active_indices.append(global_idx)

            del out, generated_ids, batched_hidden_states
            active_indices = next_active_indices
        
        del batch_sam_feats, batch_vid_lens
        
        _clean_memory()

        return (out_masks, final_states, out_metadata, out_keyframe_masks, out_num_objects, out_keyframe_indices, out_keyframe_tensors)

    def forward_train(
        self,
        processor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        sam_images = kwargs.pop("sam_images", None)
        keyframe_pixel_tensors = kwargs.pop("keyframe_pixel_tensors", None)
        seg_counts_list = kwargs.pop("seg_counts_list", None)

        self.seg_token_idx = processor.tokenizer.convert_tokens_to_ids('[SEG]')

        # 2. LLM Forward Pass
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )

        all_video_masks = []
        all_keyframe_masks = []

        if sam_images is None and keyframe_pixel_tensors is None:
            return outputs, [[] for _ in range(len(input_ids))], [[] for _ in range(len(input_ids))]

        hidden_states = outputs.hidden_states[-1]
        hidden_states = self.text_hidden_fcs(hidden_states)
        
        total_bs = input_ids.shape[0]
        MAX_OBJS = self.max_turn

        # =================================================================
        # =================================================================
        common_video_feats = None
        common_video_len = 0
        
        if sam_images is not None and len(sam_images) > 0 and sam_images[0] is not None:
            first_video_tensor = sam_images[0]
            common_video_len = first_video_tensor.shape[0]
            
            with torch.no_grad():
                common_video_feats = self.grounding_encoder.get_sam2_forward_images(first_video_tensor)

        for i in range(total_bs):
            curr_hidden = hidden_states[i]     
            curr_input_ids = input_ids[i]
            
            seg_mask = (curr_input_ids == self.seg_token_idx)
            seg_indices = seg_mask.nonzero(as_tuple=True)[0]
            num_objs = len(seg_indices)

            video_masks_per_sample = []
            keyframe_masks_per_sample = []

            if num_objs == 0:
                all_video_masks.append([])
                all_keyframe_masks.append([])
                continue
            
            process_num = min(num_objs, MAX_OBJS) 
            all_obj_embs = curr_hidden[seg_indices][:process_num]

            # =================================================================
            # =================================================================
            if common_video_feats is not None:
                # 1. Expand features for process_num objects
                with torch.no_grad():
                    base_sam_states = self.grounding_encoder.get_sam2_embeddings_expand(
                        common_video_feats, expand_size=process_num
                    )

                # 2. Decoder Prompt Prep
                pred_embeddings_video = self.generate_video_pred_embeddings([all_obj_embs], [common_video_len])
                language_embeddings = torch.cat(pred_embeddings_video, dim=0)[:, None]
                
                # 3. SAM2 Decoder
                pred_vid_mask = self.grounding_encoder.inject_language_embd_nf_nobj(
                    base_sam_states, language_embeddings, nf_nobj=(common_video_len, process_num)
                ) # Output: [Total_Time, N_Obj, H, W]

                if pred_vid_mask is not None:
                    for obj_idx in range(process_num):
                        video_masks_per_sample.append(pred_vid_mask[:, obj_idx, :, :])
                else:
                    video_masks_per_sample = [None] * process_num
                
                del base_sam_states, language_embeddings, pred_vid_mask
            else:
                 video_masks_per_sample = [None] * process_num

            # =================================================================
            # =================================================================
            curr_kfs = keyframe_pixel_tensors[i] if (keyframe_pixel_tensors and i < len(keyframe_pixel_tensors)) else []
            curr_counts = seg_counts_list[i] if (seg_counts_list and i < len(seg_counts_list)) else []
            
            processed_global_obj_count = 0

            for t_idx, count in enumerate(curr_counts):
                turn_masks_stack = None

                if count > 0 and processed_global_obj_count < process_num:
                    
                    current_process_count = min(count, process_num - processed_global_obj_count)
                    
                    if t_idx < len(curr_kfs):
                        kf_tensor = curr_kfs[t_idx].unsqueeze(0).to(dtype=curr_hidden.dtype, device=curr_hidden.device)
                        
                        turn_prompts = all_obj_embs[processed_global_obj_count : processed_global_obj_count + current_process_count]

                        # 3. SAM2 Forward (Encoder)
                        kf_feat = self.grounding_encoder.get_sam2_forward_images(kf_tensor)
                        
                        # Expand features for N objects in this turn
                        kf_states = self.grounding_encoder.get_sam2_embeddings_expand(
                            kf_feat, expand_size=current_process_count
                        )
                        
                        # Prepare Language Prompts
                        kf_lang_list = self.generate_video_pred_embeddings([turn_prompts], [1])
                        kf_lang_emb = torch.cat(kf_lang_list, dim=0)[:, None]

                        # 4. SAM2 Decoder
                        # Output: [1, N_local, H, W]
                        kf_pred_mask = self.grounding_encoder.inject_language_embd_nf_nobj(
                            kf_states, kf_lang_emb, nf_nobj=(1, current_process_count)
                        )
                        
                        if kf_pred_mask is not None:
                            turn_masks_stack = kf_pred_mask.squeeze(0) 
                        
                        del kf_feat, kf_states, kf_pred_mask, kf_lang_emb, kf_tensor
                    
                    processed_global_obj_count += current_process_count
                
                keyframe_masks_per_sample.append(turn_masks_stack)

            while len(video_masks_per_sample) < num_objs: 
                video_masks_per_sample.append(None)

            all_video_masks.append(video_masks_per_sample)
            all_keyframe_masks.append(keyframe_masks_per_sample)
            
        del common_video_feats

        gc.collect()
        torch.cuda.empty_cache()

        return outputs, all_video_masks, all_keyframe_masks

    def get_sam_embedding(self, hidden_states, if_detach_res_loss=False):
        query_hidden_state = hidden_states[:, -self.model_num_of_query:]

        if if_detach_res_loss:
            query_hidden_state = query_hidden_state.detach()

        if self.model_if_use_qwen_connector:
            query_hidden_state = self.connector(query_hidden_state)

        query_hidden_state = self.conv_1d(query_hidden_state.transpose(1, 2)).transpose(1, 2).contiguous()
        sam_embedding = self.proj_to_sam(query_hidden_state)
         
        return sam_embedding

    def get_seg_hidden_states(self, hidden_states, output_ids, seg_id):
        seg_mask = output_ids == seg_id
        n_out = len(seg_mask)
        return hidden_states[-n_out:][seg_mask]
    
    def postprocess_masks(self, masks, orig_hw):
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

    def generate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        assert len(pred_embeddings_list) == len(frames_per_batch)
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video
