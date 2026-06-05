import json
import os
import deepspeed
import PIL
import torch
import transformers
import torch.distributed as dist
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from torch.utils.data import Sampler
from accelerate.utils import gather, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.grpo_config import GRPOConfig
from qwen_vl_utils import process_vision_info
from projects.open_r1.trainer.utils import pad
from projects.open_r1.arguments import GRPOScriptArguments
import torch.nn.functional as F
from projects.open_r1.trainer.sam_loss import dice_loss, sigmoid_bce_loss, calculate_boundary_iou
from PIL import Image, ImageDraw, ImageFont

from projects.open_r1.trainer.sa2va.modeling_sa2va_cot import Sa2VAChatModelQwenCOTV2

import logging
logger = logging.getLogger(__name__)
import time
import numpy as np

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
local_rank = int(os.environ.get("LOCAL_RANK", -1))
import gc
from pycocotools import mask as maskUtils


def _clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Data Collator
def data_collator(features):
    return features

class HomogeneousRepeatSampler(Sampler):
    """
    Main behavior:
    1. Preserve dataset grouping by iterating through each child dataset separately.
    2. Shuffle within each child dataset.
    3. Repeat each sample repeat_count times for GRPO.
    """
    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
        if hasattr(data_source, 'lens'):
            self.lengths = data_source.lens
        else:
            self.lengths = [len(data_source)]

    def __iter__(self):
        final_indices = []
        current_offset = 0
        
        for length in self.lengths:
            if length > 0:
                perm = torch.randperm(length, generator=self.generator)
                perm += current_offset
                final_indices.extend(perm.tolist())
            
            current_offset += length
        
        expanded_indices = [
            idx 
            for idx in final_indices 
            for _ in range(self.repeat_count)
        ]
        
        return iter(expanded_indices)

    def __len__(self):
        return len(self.data_source) * self.repeat_count

class RepeatRandomSampler(Sampler):
    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

class RepeatSampler(Sampler):
    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in range(self.num_samples)
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count
    
def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    response_length = max(len(sub_list) for sub_list in response)
    target_length = max_length if max_length is not None and max_length > response_length else response_length
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor


class Qwen3VLGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                None, None),
        peft_config = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        script_args: GRPOScriptArguments = None, 
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.vis = script_args.if_use_visualization
        output_dir = os.path.join(args.output_dir, "logs", "info")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        self.rec_loss_ratio = script_args.rec_loss_ratio
        self.res_loss_ratio = script_args.res_loss_ratio
        self.if_detach_res_loss = script_args.if_detach_res_loss
        
        self.if_freeze_llm = script_args.if_freeze_llm
        self.if_use_pixel_reward = script_args.if_use_pixel_reward

        self.script_args = script_args
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        
        # Model Initialization
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype", "auto")
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                model_init_kwargs["torch_dtype"] = torch.bfloat16

            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
        
            model_init_kwargs.pop("use_cache", None)
            model = Sa2VAChatModelQwenCOTV2.from_pretrained(model, **model_init_kwargs)
            # Freeze/Unfreeze logic
            model.grounding_encoder.requires_grad_(False)
            model.grounding_encoder.sam2_model.sam_prompt_encoder.requires_grad_(True)
            model.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)
            
            if self.if_freeze_llm:
                model.visual.requires_grad = False
                model.model.requires_grad = False
                model.lm_head.requires_grad = False
        else:
            model_id = model.config._name_or_path

        model.max_turn = script_args.max_turn

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("\n" + "="*20 + " Trainable Layers " + "="*20)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)
            print("="*60 + "\n")

        # Ref Model
        if is_deepspeed_zero3_enabled():
            self.ref_model = Sa2VAChatModelQwenCOTV2.from_pretrained(model_id, **model_init_kwargs).model
        elif peft_config is None:
            self.ref_model = Sa2VAChatModelQwenCOTV2.from_pretrained(model_id, **model_init_kwargs).model
        else:
            self.ref_model = None

        # Processor
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)
            processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            processing_class.image_processor.max_pixels = max_pixels
            processing_class.image_processor.min_pixels = min_pixels

        # Rewards
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        set_seed(args.seed, device_specific=True)
        self.accelerator.wait_for_everyone()

    def _get_train_sampler(self, dataset) -> Sampler:
        # return HomogeneousRepeatSampler(self.train_dataset, self.num_generations, seed=self.args.seed)
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)
    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _decode_single_frame(self, object_masks_list, t, image_size):
        """
        Helper: decode a single-frame mask from the original RLE list by frame index t.
        This mirrors dataset decode_mask logic, but decodes only one frame.
        """
        current_frame_masks = []
        
        for object_masks in object_masks_list:
            mask = np.zeros(image_size, dtype=np.uint8)
            
            if len(object_masks) > 0:
                for i_anno in range(len(object_masks)):
                    if t < len(object_masks[i_anno]):
                        rle = object_masks[i_anno][t]
                        if rle is not None:
                            m = maskUtils.decode(rle)
                            if m.ndim == 3: m = m.sum(axis=2)
                            mask = mask | m.astype(np.uint8)
            
            current_frame_masks.append(mask)
            
        # Stack -> [N_obj, H, W]
        ret = np.stack(current_frame_masks, axis=0)
        
        if ret.shape[0] == 1:
            ret = ret.squeeze(0)
            
        return ret

    def _get_per_token_logps(self, model, **inputs):
        original_input_ids = inputs['input_ids']

        if "keyframe_pixel_tensors" not in inputs and "sam_images" not in inputs:
            model_output = model.forward(**inputs)
            low_res_masks = [None] * len(inputs["input_ids"])
            keyframe_masks = [None] * len(inputs["input_ids"])
        else:
            model_output, low_res_masks, keyframe_masks = model.forward_train(processor=self.processing_class, **inputs)

        logits = model_output.logits[:, :-1, :]
        input_ids = original_input_ids[:, 1:]

        del model_output 

        log_probs = logits.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(2)).squeeze(2)

        del logits
        _clean_memory()

        return token_log_prob, low_res_masks, keyframe_masks

    def _get_per_token_logps_lowmem(self, model, **inputs):
        original_input_ids = inputs['input_ids']
        
        if "keyframe_pixel_tensors" not in inputs and "sam_images" not in inputs:
            model_output = model.forward(**inputs)
            low_res_masks = [None] * len(inputs["input_ids"])
            keyframe_masks = [None] * len(inputs["input_ids"])
        else:
            model_output, low_res_masks, keyframe_masks = model.forward_train(processor=self.processing_class, **inputs)

        logits = model_output.logits[:, :-1, :]
        input_ids = original_input_ids[:, 1:]
        del model_output 
        _clean_memory()

        chunk_size = 1024
        token_log_probs = []
        entropies = []
        
        for i in range(0, logits.shape[1], chunk_size):
            chunk_logits = logits[:, i:i+chunk_size, :].float()
            
            log_p = F.log_softmax(chunk_logits, dim=-1)
            
            # Token LogProb
            chunk_ids = input_ids[:, i:i+chunk_size]
            t_lp = torch.gather(log_p, dim=2, index=chunk_ids.unsqueeze(2)).squeeze(2)
            token_log_probs.append(t_lp)
            
            # Entropy: -sum(exp(log_p) * log_p)
            ent = -torch.sum(torch.exp(log_p) * log_p, dim=-1)
            entropies.append(ent)
            
            del chunk_logits, log_p
            _clean_memory()
            
        token_log_prob = torch.cat(token_log_probs, dim=1)
        entropy = torch.cat(entropies, dim=1)
        
        del logits
        _clean_memory()
        return token_log_prob, entropy, low_res_masks, keyframe_masks

    def _get_per_mask_logps_cot(self, forward_mask_list, rollout_mask_list, seg_counts_list):
        """
        Compute mask log probabilities and flatten them to align with [SEG] tokens.
        
        Args:
            forward_mask_list: [Batch, Turns, Tensor(N_obj, H, W) or None] from forward_train.
            rollout_mask_list: [Batch, Turns, Tensor(N_obj, H, W) or None] from generation.
            seg_counts_list: [Batch, Turns], number of generated objects per turn.
            
        Returns:
            all_sample_mask_logps: List[List[Tensor]]。
            Outer list is batch-level; inner list length equals the total [SEG] token count for that sample.
        """
        target_size = (32, 32)
        all_sample_mask_logps = []
        
        device = self.accelerator.device

        for i, (f_turns, r_turns, counts) in enumerate(zip(forward_mask_list, rollout_mask_list, seg_counts_list)):
            current_sample_flat_logps = []
            
            num_turns = len(counts)
            
            if len(f_turns) < num_turns or len(r_turns) < num_turns:
                 if self.accelerator.is_main_process:
                    print(f"\n[Warning] Batch {i} Turn Mismatch! Counts: {len(counts)}, Forward: {len(f_turns)}, Rollout: {len(r_turns)}")

            for t, count in enumerate(counts):
                if count == 0:
                    continue
                
                f_mask_tensor = f_turns[t] if t < len(f_turns) else None
                r_mask_tensor = r_turns[t] if t < len(r_turns) else None
                
                error_flag = False
                if f_mask_tensor is None or r_mask_tensor is None:
                    error_flag = True
                    if self.accelerator.is_main_process:
                        print(f"[Error] Batch {i} Turn {t}: Missing mask tensor. Count={count}, "
                              f"F_exist={f_mask_tensor is not None}, R_exist={r_mask_tensor is not None}")
                
                elif f_mask_tensor.shape[0] != count or r_mask_tensor.shape[0] != count:
                    error_flag = True
                    if self.accelerator.is_main_process:
                        print(f"[Error] Batch {i} Turn {t}: Dimension mismatch with Count. "
                              f"Count={count}, Forward={f_mask_tensor.shape}, Rollout={r_mask_tensor.shape}")

                if error_flag:
                    for _ in range(count):
                        dummy_logp = torch.tensor(0.0, device=device, requires_grad=True)
                        current_sample_flat_logps.append(dummy_logp)
                    continue

                # f_mask_tensor: [Count, H, W]
                # r_mask_tensor: [Count, H, W]
                
                with torch.no_grad():
                    r_resized = F.interpolate(r_mask_tensor.unsqueeze(1).float(), size=target_size, mode="bilinear", align_corners=False)
                    r_binary = (r_resized > 0.5).float() # [Count, 1, 32, 32]

                if f_mask_tensor.dim() == 3:
                    f_input = f_mask_tensor.unsqueeze(1) # [Count, 1, H, W]
                else:
                    f_input = f_mask_tensor # Should be [Count, 1, H, W] usually
                
                f_resized = F.interpolate(f_input.float(), size=target_size, mode="bilinear", align_corners=False)

                pixel_neg_bce = -F.binary_cross_entropy_with_logits(f_resized, r_binary, reduction='none')
                
                obj_log_probs = pixel_neg_bce.mean(dim=(1, 2, 3)) 
                
                for k in range(count):
                    current_sample_flat_logps.append(obj_log_probs[k])

            all_sample_mask_logps.append(current_sample_flat_logps)

        return all_sample_mask_logps

    def _get_per_mask_logps_cot_topk(self, forward_mask_list, rollout_mask_list):
        """
        Compute mask log probabilities.
        Implementation notes:
        1. Aggregate logits over the full time/frame dimension per [SEG] token.
        2. Use Top-K/OHEM over the hardest pixels across frames for each [SEG] token.
        """
        target_size = (32, 32)
        all_sample_mask_logps = []
        
        topk_ratio = 1.0

        for i, (f_turns, r_turns) in enumerate(zip(forward_mask_list, rollout_mask_list)):
            current_sample_logps = []
            
            if len(f_turns) != len(r_turns):
                logger.warning(f"Sample {i} mask mismatch: f_turns: {len(f_turns)}, r_turns: {len(r_turns)}")

            num_common = min(len(f_turns), len(r_turns))
            
            for k in range(num_common):
                f_mask = f_turns[k] 
                r_mask = r_turns[k] 
                
                if f_mask is None or r_mask is None or f_mask.numel() == 0:
                    dummy_logp = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
                    current_sample_logps.append(dummy_logp)
                    continue
                
                if f_mask.dim() == 3: f_mask = f_mask.unsqueeze(1) 
                if r_mask.dim() == 3: r_mask = r_mask.unsqueeze(1)

                f_mask = f_mask.float()
                r_mask = r_mask.float()

                f_resized = F.interpolate(f_mask, size=target_size, mode="bilinear", align_corners=False)
                
                with torch.no_grad():
                    r_resized = F.interpolate(r_mask, size=target_size, mode="bilinear", align_corners=False)
                    r_binary = (r_resized > 0.5).float()

                
                # shape: [T, 1, 32, 32]
                pixel_log_probs_grid = -F.binary_cross_entropy_with_logits(
                    f_resized, 
                    r_binary, 
                    reduction='none'
                )
                
                pixel_log_probs_flat = pixel_log_probs_grid.view(-1)
                
                # 3. Top-K Selection (OHEM)
                num_pixels = pixel_log_probs_flat.numel()
                K = max(1, int(num_pixels * topk_ratio))
                
                topk_log_probs, _ = torch.topk(pixel_log_probs_flat, K, largest=False)

                seg_token_log_prob = topk_log_probs.sum()
                
                current_sample_logps.append(seg_token_log_prob)
            
            all_sample_mask_logps.append(current_sample_logps)

        return all_sample_mask_logps

    def _prepare_sam_inputs(self, inputs):
        """Helper to extract and stack SAM images."""
        sam_images_list = [x.get("sam_image") for x in inputs]
        valid_sam_images = [img for img in sam_images_list if img is not None]
        
        if not valid_sam_images:
            return None
            
        processed_sam_images = []
        for img in sam_images_list:
            if img is None: continue 
            if len(img.shape) == 3: 
                processed_sam_images.append(img.unsqueeze(0).unsqueeze(0))
            elif len(img.shape) == 4: 
                    processed_sam_images.append(img.unsqueeze(0))
            else:
                    processed_sam_images.append(img)
        
        return torch.cat(processed_sam_images, dim=0).to(self.accelerator.device)

    def _finalize_batch_outputs(self, final_input_states):
        device = self.accelerator.device
        
        batch_msgs = [s["messages"] for s in final_input_states]
        batch_imgs = []
        for s in final_input_states:
            images = s["images"] if (s["images"] and len(s["images"]) > 0) else None
            tmp_images = s["tmporal_images"] if (s["tmporal_images"] and len(s["tmporal_images"]) > 0) else []
            if images is None: batch_imgs.append(None)
            else: batch_imgs.append(tmp_images+images)

        batch_texts = [
            self.processing_class.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch_msgs
        ]

        batched_inputs = self.processing_class(
            text=batch_texts,
            images=batch_imgs,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False
        ).to(device)

        input_ids = batched_inputs["input_ids"]
        attention_mask = batched_inputs["attention_mask"] 
        
        completion_mask = torch.zeros_like(input_ids, dtype=torch.int)
        all_turn_completion_ids = [] 

        for i, (msgs, imgs) in enumerate(zip(batch_msgs, batch_imgs)):
            current_len = 0
            img_ptr = 0
            temp_msgs = []
            temp_imgs = []
            sample_turns = [] 

            for msg in msgs:
                temp_msgs.append(msg)
                if msg['role'] == 'user' and isinstance(msg['content'], list):
                    img_count = sum(1 for x in msg['content'] if x['type'] == 'image')
                    if imgs:
                        temp_imgs.extend(imgs[img_ptr : img_ptr + img_count])
                        img_ptr += img_count
                
                need_gen_prompt = (msg['role'] == 'user')
                
                temp_text = self.processing_class.apply_chat_template(
                    temp_msgs, 
                    tokenize=False, 
                    add_generation_prompt=need_gen_prompt
                )
                
                safe_temp_imgs = temp_imgs if temp_imgs else None
                
                temp_out = self.processing_class(
                    text=[temp_text],
                    images=safe_temp_imgs,
                    return_tensors="pt",
                    padding=False
                )
                new_len = temp_out.input_ids.shape[1]
                
                if msg['role'] == 'assistant':
                    start = current_len
                    end = new_len
                    if end > start and start < input_ids.shape[1]:
                        real_end = min(end, input_ids.shape[1])
                        completion_mask[i, start:real_end] = 1
                        sample_turns.append(input_ids[i, start:real_end])
                
                current_len = new_len
                
            all_turn_completion_ids.append(sample_turns)

        completion_mask = completion_mask * attention_mask
        
        return batched_inputs, input_ids, all_turn_completion_ids, completion_mask

    def _generate_completions_cot(self, model, inputs):
        if hasattr(self.processing_class, "tokenizer"):
            self.processing_class.tokenizer.padding_side = "left"

        sam_images_tensor = self._prepare_sam_inputs(inputs)

        model_sa2va = self.accelerator.unwrap_model(model)
        unwrapped_model = model_sa2va.model if hasattr(model_sa2va, "model") else model_sa2va
        
        is_gradient_checkpointing_enabled = getattr(unwrapped_model, "is_gradient_checkpointing", False) or \
                                          getattr(unwrapped_model.config, "gradient_checkpointing", False)
        
        if is_gradient_checkpointing_enabled:
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.gradient_checkpointing = False
        
        unwrapped_model.config.use_cache = True

        with torch.inference_mode():
            with torch.autocast(device_type=self.accelerator.device.type, dtype=torch.bfloat16):
                
                generation_kwargs = {
                    "max_new_tokens": self.max_completion_length,
                    "do_sample": True,
                    "temperature": self.args.temperature,
                    "top_p": 0.9,
                    "use_cache": True,
                    "pad_token_id": self.processing_class.pad_token_id,
                    "eos_token_id": self.processing_class.eos_token_id,
                }
                
                (ret_masks, final_input_states, all_turn_metadata, out_keyframe_masks, 
                 out_num_objects, out_keyframe_indices, out_keyframe_tensors) = \
                    model_sa2va.generation_forward_cot(
                        processor=self.processing_class,
                        raw_inputs_list=[x.copy() for x in inputs],
                        sam_images_tensor=sam_images_tensor,
                        **generation_kwargs
                    )

        unwrapped_model.config.use_cache = False
        if is_gradient_checkpointing_enabled:
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.gradient_checkpointing = True

        batched_inputs, prompt_completion_ids, completion_ids, completion_mask = self._finalize_batch_outputs(final_input_states)

        return (batched_inputs, 
                prompt_completion_ids, 
                completion_ids, 
                completion_mask[:, 1:], 
                ret_masks,
                all_turn_metadata,
                out_keyframe_masks,
                out_num_objects,
                out_keyframe_indices,
                out_keyframe_tensors)

    def _compute_video_segmentation_loss(self, inputs, pred_masks_list, out_num_objects):
        """
        Compute loss and IoU for video temporal masks.
        Used to evaluate temporal masks generated during generation when available.
        The target is inputs['mask'], usually downsampled temporal ground truth.
        """
        mask_bce_loss = 0.0
        mask_dice_loss = 0.0
        all_sample_ious = []
        total_mask_pairs = 0
        
        for i, (flat_masks, turn_counts) in enumerate(zip(pred_masks_list, out_num_objects)):
            gt_mask_raw = inputs[i].get("mask") # [T, H, W]
            current_sample_ious = []
            
            if gt_mask_raw is None:
                for _ in turn_counts:
                    current_sample_ious.append([])
                all_sample_ious.append(current_sample_ious)
                continue

            if isinstance(gt_mask_raw, np.ndarray):
                gt_mask = torch.from_numpy(gt_mask_raw).to(self.accelerator.device).float()
            else:
                gt_mask = gt_mask_raw.float().to(self.accelerator.device)

            if gt_mask.dim() == 2: gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
            elif gt_mask.dim() == 3: gt_mask = gt_mask.unsqueeze(1) # [T, 1, H, W]
            
            cursor = 0
            
            for count in turn_counts:
                if count == 0:
                    current_sample_ious.append([0.0] * gt_mask.shape[0])
                    continue
                
                current_turn_masks = flat_masks[cursor : cursor + count]
                cursor += count
                
                valid_masks = [m for m in current_turn_masks if m is not None]
                if not valid_masks:
                    current_sample_ious.append([0.0] * gt_mask.shape[0])
                    continue

                # Stack: [Num_Obj, T, H, W]
                pred_stack = torch.stack(valid_masks)
                
                N_obj, T_pred, H_pred, W_pred = pred_stack.shape
                pred_input = pred_stack.view(-1, 1, H_pred, W_pred).float()
                
                pred_resized_flat = F.interpolate(
                    pred_input, 
                    size=gt_mask_raw.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
                pred_final = pred_resized_flat.view(N_obj, T_pred, 1, gt_mask_raw.shape[-2], gt_mask_raw.shape[-1])
                
                curr_gt = gt_mask
                if curr_gt.shape[0] == 1 and T_pred > 1:
                    curr_gt = curr_gt.expand(T_pred, -1, -1, -1)
                curr_gt = curr_gt.unsqueeze(0).expand(N_obj, -1, -1, -1, -1)

                if curr_gt.shape[1] != pred_final.shape[1]:
                    current_sample_ious.append([0.0] * T_pred)
                    continue

                flat_pred = pred_final.flatten(0, 1) 
                flat_gt = curr_gt.flatten(0, 1)

                loss_multimask = sigmoid_bce_loss(flat_pred, flat_gt, 1., loss_on_multimask=True)
                loss_multidice = dice_loss(flat_pred, flat_gt, 1., True)

                mask_bce_loss += loss_multimask.mean() * N_obj
                mask_dice_loss += loss_multidice.mean() * N_obj
                total_mask_pairs += N_obj

                with torch.no_grad():
                    temp_pred = (pred_final > 0).float()
                    temp_gt = (curr_gt > 0.5).float()
                    inter = (temp_pred * temp_gt).sum(dim=(3, 4))
                    union = temp_pred.sum(dim=(3, 4)) + temp_gt.sum(dim=(3, 4)) - inter
                    iou_map = inter / (union + 1e-6)
                    iou_map[union < 1] = 1.0
                    avg_iou_per_frame = iou_map.squeeze(2).mean(dim=0)
                    current_sample_ious.append(avg_iou_per_frame.cpu().tolist())

            all_sample_ious.append(current_sample_ious)

        if total_mask_pairs > 0:
            mask_bce_loss /= total_mask_pairs
            mask_dice_loss /= total_mask_pairs
        else:
            mask_bce_loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
            mask_dice_loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        return mask_bce_loss + mask_dice_loss, all_sample_ious

    def _compute_keyframe_segmentation_loss(self, inputs, keyframe_masks_list, keyframe_indices_list):
        """
        Optimized keyframe loss computation with on-the-fly ground-truth decoding.
        """
        loss_bce = 0.0
        loss_dice = 0.0
        total_valid_frames = 0
        all_ious = []

        for i, (turn_masks, turn_indices) in enumerate(zip(keyframe_masks_list, keyframe_indices_list)):
            sample_ious = []
            
            gt_mask_full_raw = inputs[i].get('mask_full')
            img_size = inputs[i].get('image_size')        # (H, W)
            total_frames = inputs[i].get('total_frames')
            
            if gt_mask_full_raw is None or img_size is None:
                all_ious.append([0.0] * len(turn_masks))
                continue
            
            if not isinstance(img_size, (tuple, list)): 
                img_size = (img_size[0].item(), img_size[1].item())
            elif isinstance(img_size, list):
                img_size = tuple(img_size)

            if total_frames is not None:
                if isinstance(total_frames, torch.Tensor): 
                    T_limit = total_frames.item()
                else:
                    T_limit = total_frames
            else:
                T_limit = 99999

            num_turns = min(len(turn_masks), len(turn_indices))
            
            for t in range(num_turns):
                pred_mask = turn_masks[t]
                k_idx = turn_indices[t]
                
                if pred_mask is None or k_idx is None or k_idx < 0:
                    sample_ious.append(0.0)
                    continue
                
                safe_idx = int(min(k_idx, T_limit - 1))
                
                try:
                    gt_slice_raw = self._decode_single_frame(gt_mask_full_raw, safe_idx, img_size)
                except Exception as e:
                    print(f"Decode error at batch {i} frame {safe_idx}: {e}")
                    sample_ious.append(0.0)
                    continue

                if isinstance(gt_slice_raw, np.ndarray):
                    gt_slice = torch.from_numpy(gt_slice_raw).to(self.accelerator.device).float()
                else:
                    gt_slice = gt_slice_raw.to(self.accelerator.device).float()
                
                if gt_slice.dim() == 2: 
                    gt_slice = gt_slice.unsqueeze(0).unsqueeze(0)
                elif gt_slice.dim() == 3: 
                    gt_slice = gt_slice.unsqueeze(0)

                pred_input = pred_mask.float()
                if pred_input.dim() == 3 and pred_input.shape[0] > 1:
                    pred_input = pred_input[-1:]

                if pred_input.dim() == 2: 
                    pred_input = pred_input.unsqueeze(0).unsqueeze(0)
                elif pred_input.dim() == 3: 
                    pred_input = pred_input.unsqueeze(0)
                
                target_h, target_w = gt_slice.shape[-2:]
                pred_resized = F.interpolate(
                    pred_input, 
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                
                l_bce = sigmoid_bce_loss(pred_resized, gt_slice, 1.0, loss_on_multimask=True)
                l_dice = dice_loss(pred_resized, gt_slice, 1.0, True)
                
                loss_bce += l_bce.mean()
                loss_dice += l_dice.mean()
                total_valid_frames += 1
                
                with torch.no_grad():
                    pred_bin = (pred_resized > 0).float()
                    gt_bin = (gt_slice > 0.5).float()
                    
                    inter = (pred_bin * gt_bin).sum()
                    union = pred_bin.sum() + gt_bin.sum() - inter
                    iou = (inter / (union + 1e-6)).item()
                    sample_ious.append(iou)
            
            all_ious.append(sample_ious)
            
        if total_valid_frames > 0:
            avg_loss = (loss_bce + loss_dice) / total_valid_frames
        else:
            avg_loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
            
        return avg_loss, all_ious

    def _compute_rewards(self, inputs, completion_ids, segmentation_ious, keyframe_ious=None):
        # inputs_ = [x for x in inputs for _ in range(self.num_generations)]
        prompts = [x["prompt"] for x in inputs]
        completions = [self.processing_class.batch_decode(gen_n_completion, skip_special_tokens=self.script_args.skip_special_tokens) for gen_n_completion in completion_ids]
                
        completions_struct = completions

        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        for i, reward_func in enumerate(self.reward_funcs):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            reward_kwargs['current_step'] = self.state.global_step
            reward_kwargs['per_round_maskious'] = segmentation_ious
            reward_kwargs['per_round_keyframe_ious'] = keyframe_ious # Pass to Reward Func
            
            output_reward_func = reward_func(prompts=prompts, completions=completions_struct, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        return rewards_per_func

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # 1. Generation
        total_start_time = time.perf_counter()
        t_gen_start = time.perf_counter()
        (batched_inputs, prompt_completion_ids, completion_ids, completion_mask, 
            low_res_masks_generation, all_turn_metadata, keyframe_masks_generation, 
            out_num_objects, out_keyframe_indices, out_keyframe_tensors) = self._generate_completions_cot(model, inputs)
        time_generation = time.perf_counter() - t_gen_start

        # 2. Forward Pass (Train Model)
        batched_inputs_train = batched_inputs.copy()
        batched_inputs_train["keyframe_pixel_tensors"] = out_keyframe_tensors
        batched_inputs_train["seg_counts_list"] = out_num_objects
        batched_inputs_train["sam_images"] = self._prepare_sam_inputs(inputs)
        
        model_ = model.module if hasattr(model, "module") else model
        per_token_logps, vid_masks_fwd, kf_masks_fwd = self._get_per_token_logps(model_, **batched_inputs_train)

        per_mask_logps_list = []
        if self.script_args.use_mask_logps:
            # per_mask_logps_list = self._get_per_mask_logps_cot_topk(low_res_masks_forward, low_res_masks_generation)
            per_mask_logps_list = self._get_per_mask_logps_cot(kf_masks_fwd, keyframe_masks_generation, out_num_objects)
            
        batched_inputs_train.pop("keyframe_pixel_tensors", None)
        batched_inputs_train.pop("seg_counts_list", None)
        batched_inputs_train.pop("sam_images", None)
        
        # For reward iou computation
        _, segmentation_ious_generation = self._compute_video_segmentation_loss(inputs, low_res_masks_generation, out_num_objects)
        _, keyframe_ious_generation = self._compute_keyframe_segmentation_loss(inputs, keyframe_masks_generation, out_keyframe_indices)
        
        loss_vid_aux, _ = self._compute_video_segmentation_loss(inputs, vid_masks_fwd, out_num_objects)
        loss_kf_aux, _ = self._compute_keyframe_segmentation_loss(inputs, kf_masks_fwd, out_keyframe_indices)
        res_loss_forward = loss_vid_aux + loss_kf_aux

        # 4. Metrics & Alignment
        ppl = -((per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # 5. Reference Model Forward
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps, _, _ = self._get_per_token_logps(self.ref_model, **batched_inputs_train)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps, _, _ = self._get_per_token_logps(model, **batched_inputs_train)

        ref_ppl = -((ref_per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # 6. KL Divergence
        k1 = per_token_logps - ref_per_token_logps
        k3 = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        kimi = 0.5 * (per_token_logps - ref_per_token_logps)**2
        
        if self.script_args.kl_approximator == 'k3': per_token_kl = k3
        elif self.script_args.kl_approximator == 'k1': per_token_kl = k1
        elif self.script_args.kl_approximator in ['kimikl', 'fullkimi']: per_token_kl = kimi

        # 7. Rewards (Update to pass keyframe ious)
        rewards_per_func_ = self._compute_rewards(inputs, completion_ids, segmentation_ious_generation, keyframe_ious_generation)
        rewards_per_func = gather(rewards_per_func_)
        rewards = rewards_per_func.sum(dim=1) * self.script_args.reward_scale
        rewards_this = rewards_per_func_.sum(dim=1).detach() * self.script_args.reward_scale

        # 8. Advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        
        if self.script_args.kl_approximator == 'fullkimi':
            advantages = rewards - mean_grouped_rewards
        else:   
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        advantages = advantages[process_slice]
            
        if self.script_args.use_mask_logps and per_mask_logps_list:
            seg_id = self.processing_class.tokenizer.convert_tokens_to_ids("[SEG]")
            target_ids_batch = prompt_completion_ids[:, 1:]
            
            for i, (row_mask_logps, row_ids) in enumerate(zip(per_mask_logps_list, target_ids_batch)):
                if not row_mask_logps: continue
                seg_idxs = (row_ids == seg_id).nonzero(as_tuple=True)[0]
                
                if len(row_mask_logps) != len(seg_idxs):
                    if self.accelerator.is_main_process:
                        print(f"\n[Fatal Error] Batch {i}: Mask Logps count ({len(row_mask_logps)}) "
                              f"!= SEG Token count ({len(seg_idxs)}). Skipping injection to avoid crash.")
                    continue
                mask_logps_tensor = torch.stack(row_mask_logps).to(per_token_logps.device)
                target_indices = seg_idxs
                
                per_token_logps[i, target_indices] += mask_logps_tensor
            
        # 9. Policy Loss
        if self.script_args.kl_approximator == 'fullkimi':
            per_token_loss = -torch.exp(per_token_logps) * advantages.unsqueeze(1)
        else:
            per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)

        if self.script_args.use_kl: per_token_loss += self.beta * per_token_kl
        
        loss_llm_rl = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = loss_llm_rl

        # 10. Logging
        local_completion_lens = completion_mask.sum(dim=1).float()
        all_completion_lens = self.accelerator.gather_for_metrics(local_completion_lens)
        self._metrics["completion_length"].append(all_completion_lens.mean().item())

        local_turn_counts = torch.tensor(
            [len(meta) for meta in all_turn_metadata], 
            dtype=torch.float, 
            device=self.accelerator.device
        )
        all_turn_counts = self.accelerator.gather_for_metrics(local_turn_counts)
        self._metrics["avg_turns"].append(all_turn_counts.mean().item())

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        total_end_time = time.perf_counter()
        time_total = total_end_time - total_start_time

        self._metrics["timing/generation_sec"].append(time_generation)
        self._metrics["timing/total_step_sec"].append(time_total)
        
        self._save_rollout_logs(
            inputs=inputs, 
            completion_ids=completion_ids, 
            rewards=rewards_this, 
            rewards_per_func=rewards_per_func_, 
            segmentation_ious=segmentation_ious_generation, 
            keyframe_ious=keyframe_ious_generation, 
            all_turn_metadata=all_turn_metadata, 
            out_keyframe_indices=out_keyframe_indices
        )

        if self.state.global_step % self.script_args.vis_interval == 0 and self.vis:
             self._visualize_samples(
                inputs=inputs, 
                keyframe_masks_list=keyframe_masks_generation, 
                keyframe_ious=keyframe_ious_generation, 
                out_keyframe_indices=out_keyframe_indices,
                low_res_masks_list=low_res_masks_generation, 
                out_num_objects=out_num_objects,
                segmentation_ious=segmentation_ious_generation,
                all_turn_metadata=all_turn_metadata
            )

        return loss * self.rec_loss_ratio + res_loss_forward * self.res_loss_ratio
    
    def _save_rollout_logs(self, inputs, completion_ids, rewards, rewards_per_func, 
                           segmentation_ious, keyframe_ious, 
                           all_turn_metadata, out_keyframe_indices):
        """
        Save plain-text logs.
        Notes:
        1. Decode each turn tensor separately to avoid decode errors.
        2. Show the assistant response directly after each turn user query.
        """

        rewards = rewards.detach().cpu().tolist()
        rewards_pf = rewards_per_func.detach().cpu().tolist()
        
        if segmentation_ious is None: segmentation_ious = [[] for _ in range(len(inputs))]
        if keyframe_ious is None: keyframe_ious = [[] for _ in range(len(inputs))]

        reward_keys = [f.config._name_or_path.split("/")[-1] if hasattr(f, "config") else f.__name__ for f in self.reward_funcs]

        video_groups = defaultdict(list)
        for i, item in enumerate(inputs):
            paths = item.get('total_frame_path', [])
            if not paths:
                paths = item.get('image_path', [])
            
            if not paths: continue
            video_name = paths[0].split('/')[-2]
            video_groups[video_name].append(i)

        for vid_name, indices in video_groups.items():
            max_turns = 0
            for idx in indices:
                n = len(all_turn_metadata[idx]) if idx < len(all_turn_metadata) else 0
                if n > max_turns: max_turns = n
            
            group_dir = os.path.join(self.output_dir, f"step{self.state.global_step}/{vid_name}_maxT{max_turns}")
            os.makedirs(group_dir, exist_ok=True)

            group_logs = []

            for global_idx in indices:
                item = inputs[global_idx]
                problem_text = item.get('problem', 'N/A')
                
                sample_turn_tokens = completion_ids[global_idx] 
                # --- [FIX END] ---

                conversation_trace = []
                formatted_dialogue_lines = []
                
                cur_meta = all_turn_metadata[global_idx] if global_idx < len(all_turn_metadata) else []
                cur_seg_iou = segmentation_ious[global_idx] if global_idx < len(segmentation_ious) else []
                cur_kf_iou = keyframe_ious[global_idx] if global_idx < len(keyframe_ious) else []
                
                num_turns = len(cur_meta)

                for t in range(num_turns):
                    meta = cur_meta[t]
                    user_q = meta.get('user_query', 'N/A')
                    
                    # Image Info
                    if "added_images" in meta: added_imgs = meta["added_images"]
                    elif "frame_info" in meta: added_imgs = meta["frame_info"]
                    elif "temporal_indices" in meta:
                        t_inds = meta.get("temporal_indices", "[]")
                        k_ind = meta.get("keyframe_index", "None")
                        added_imgs = f"Temp Indices: {t_inds}, Keyframe: {k_ind}"
                    else: added_imgs = "N/A"

                    # IoU Handling
                    raw_iou = cur_seg_iou[t] if t < len(cur_seg_iou) else 0.0
                    kiou = cur_kf_iou[t] if t < len(cur_kf_iou) else 0.0
                    
                    if isinstance(raw_iou, list):
                        avg_iou = sum(raw_iou)/len(raw_iou) if raw_iou else 0.0
                        iou_str = f"Avg: {avg_iou:.3f} | Details: {[round(x, 2) for x in raw_iou]}"
                    else:
                        iou_str = f"{raw_iou:.3f}"
                    
                    if isinstance(kiou, list): kiou = kiou[0]

                    # Keyframe Index
                    k_idx = out_keyframe_indices[global_idx][t] if global_idx < len(out_keyframe_indices) and t < len(out_keyframe_indices[global_idx]) else 'N/A'

                    if t < len(sample_turn_tokens):
                        turn_tensor = sample_turn_tokens[t]
                        if turn_tensor.numel() > 0:
                            decoded_text = self.processing_class.decode(turn_tensor, skip_special_tokens=self.script_args.skip_special_tokens)
                        else:
                            decoded_text = "<Empty Generation>"
                    else:
                        decoded_text = "<Missing Tokens>"
                    # --- [FIX END] ---

                    conversation_trace.append({
                        "turn_idx": t, 
                        "user_query": user_q, 
                        "added_images": added_imgs,
                        "assistant_response": decoded_text,
                        "iou_raw": raw_iou, 
                        "keyframe_iou": kiou
                    })
                    
                    formatted_dialogue_lines.append(
                        f"--- Round {t} ---\n"
                        f"[User]: {user_q}\n"
                        f"[Images]: {added_imgs}\n"
                        f"[Assistant]: {decoded_text}\n"
                        f"[Metrics]: Temp IoU: {iou_str}, KF IoU: {kiou:.4f}\n"
                    )

                # Metrics
                r = rewards[global_idx]
                r_pf = rewards_pf[global_idx]
                breakdown_dict = dict(zip(reward_keys, r_pf))
                
                # Flat stats
                flat_ious = []
                if cur_seg_iou:
                    for x in cur_seg_iou:
                        if isinstance(x, list): flat_ious.extend(x)
                        else: flat_ious.append(x)
                mean_iou = sum(flat_ious) / len(flat_ious) if flat_ious else 0.0
                max_iou = max(flat_ious) if flat_ious else 0.0

                group_logs.append({
                    'step': self.state.global_step, 
                    'gen_idx': global_idx, 
                    'video_name': vid_name,
                    'problem': problem_text,
                    'turn_count': num_turns, 
                    'prompt': item.get('prompt', ''),
                    'conversation_trace': conversation_trace, 
                    'full_response_str': "\n".join(formatted_dialogue_lines),
                    'total_reward': r, 
                    'mean_mask_iou': mean_iou, 
                    'max_mask_iou': max_iou,
                    'reward_breakdown': breakdown_dict
                })

            with open(os.path.join(group_dir, "rollout_info.json"), 'w', encoding='utf-8') as f:
                json.dump({'step': self.state.global_step, 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'rollout_info': group_logs}, f, indent=2, ensure_ascii=False)
            
            with open(os.path.join(group_dir, "readable_dialogue.txt"), 'w', encoding='utf-8') as f:
                for idx, log in enumerate(group_logs):
                    rb = log['reward_breakdown']
                    rb_str = " | ".join([f"{k}: {v:.4f}" for k, v in rb.items()])
                    header = f"=== Rollout {idx} (BatchIdx: {log['gen_idx']}) (Turns: {log['turn_count']}, TotR: {log['total_reward']:.3f}, IoU: {log['mean_mask_iou']:.2f}) ==="
                    
                    f.write(f"{header}\n")
                    f.write(f"[Problem]: {log['problem']}\n")
                    f.write(f"[Rewards Detail]: {rb_str}\n") 
                    f.write(f"{log['full_response_str']}\n")
                    f.write(f"{'='*40}\n")

    def _visualize_samples(self, inputs, 
                           keyframe_masks_list, keyframe_ious, out_keyframe_indices,
                           low_res_masks_list, out_num_objects, segmentation_ious,
                           all_turn_metadata):
        """
        Enhanced visualization helper.
        
        Behavior:
        1. Input/output visualization: draw keyframe GT vs prediction and temporal mask strips by turn.
        2. Global input history: concatenate temporal frames introduced across turns into one image,
           and annotate large T{turn} F{frame_idx} labels to track the model information path.
        """
        from PIL import Image, ImageDraw, ImageFont
        import ast

        font_path = "/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/arial.ttf"

        def _get_large_font(img_h, img_w, scale=0.15):
            size = int(min(img_h, img_w) * scale)
            size = max(24, size)
            try:
                return ImageFont.truetype(font_path, size)
            except:
                return ImageFont.load_default()

        def _draw_text_with_stroke(img_np, text, color=(255, 255, 0), pos='center', scale=0.15):
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)
            W, H = pil_img.size
            font = _get_large_font(H, W, scale)
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            if pos == 'center':
                x, y = (W - text_w) // 2, (H - text_h) // 2
            elif pos == 'top_left':
                x, y = int(W * 0.02), int(H * 0.02)
            else:
                x, y = 0, 0

            stroke_width = max(2, int(font.size / 15))
            
            draw.text((x, y), text, font=font, fill=color, stroke_width=stroke_width, stroke_fill=(0,0,0))
            return np.array(pil_img)

        def _get_font(img_h, img_w, scale_factor=0.03):
            size = int(min(img_h, img_w) * scale_factor)
            size = max(20, size)
            try: 
                return ImageFont.truetype(font_path, size)
            except: 
                return ImageFont.load_default()

        def _apply_overlay(img, mask, color, alpha=0.5):
            if mask is None: return img
            if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().float().numpy()
            
            if mask.shape[-2:] != img.shape[:2][::-1]: 
                m_pil = Image.fromarray(mask)
                m_pil = m_pil.resize(img.shape[:2][::-1], resample=Image.BILINEAR)
                mask = np.array(m_pil)

            mask_bool = mask > 0.05 
            if not mask_bool.any(): return img
            
            overlay = img.copy()
            overlay[mask_bool] = color
            return (img * (1 - alpha) + overlay * alpha).astype(np.uint8)

        # 1. Group by Video Name
        video_groups = defaultdict(list)
        for idx, item in enumerate(inputs):
            paths = item.get('total_frame_path', [])
            if paths:
                v_name = paths[0].split('/')[-2]
                video_groups[v_name].append(idx)

        # 2. Iterate Groups
        for vid_name, indices in video_groups.items():
            max_turns = 0
            for idx in indices:
                n_turns = len(out_keyframe_indices[idx]) if idx < len(out_keyframe_indices) else 0
                if n_turns > max_turns: max_turns = n_turns
            
            save_dir = os.path.join(self.output_dir, f"step{self.state.global_step}/{vid_name}_maxT{max_turns}")
            os.makedirs(save_dir, exist_ok=True)

            # 3. Process Each Sample in Group
            for global_idx in indices:
                item = inputs[global_idx]
                paths = item.get('total_frame_path', [])
                if not paths: continue

                gt_mask_full_raw = item.get('mask_full') 
                img_size = item.get('image_size')
                if isinstance(img_size, list): img_size = tuple(img_size)
                
                sample_meta = all_turn_metadata[global_idx] if global_idx < len(all_turn_metadata) else []

                # =========================================================
                # [NEW] Part C: Global Cumulative Input Visualization
                # =========================================================
                global_vis_imgs = []
                
                for t, meta in enumerate(sample_meta):
                    temp_indices_str = meta.get("temporal_indices", "[]")
                    try:
                        curr_temp_indices = ast.literal_eval(temp_indices_str) if isinstance(temp_indices_str, str) else temp_indices_str
                    except: curr_temp_indices = []
                    
                    if not curr_temp_indices: continue

                    for tidx in curr_temp_indices:
                        if tidx < len(paths):
                            # Load raw image
                            raw_img = np.array(Image.open(paths[tidx]).convert('RGB'))
                            
                            img_labelled = _draw_text_with_stroke(raw_img, f"T{t}", color=(200, 200, 200), pos='top_left', scale=0.15)
                            
                            img_labelled = _draw_text_with_stroke(img_labelled, f"F{tidx}", color=(255, 255, 0), pos='center', scale=0.30)
                            
                            global_vis_imgs.append(img_labelled)
                
                if global_vis_imgs:
                    try:
                        full_history_concat = np.concatenate(global_vis_imgs, axis=1)
                        save_name = os.path.join(save_dir, f"gen{global_idx}_GLOBAL_temporal_history.jpg")
                        Image.fromarray(full_history_concat).save(save_name)
                    except Exception as e:
                        print(f"Global Vis Error {global_idx}: {e}")

                # =========================================================
                # Part A: Keyframe Visualization (Original Logic Preserved)
                # =========================================================
                kf_indices = out_keyframe_indices[global_idx]
                kf_masks_struct = keyframe_masks_list[global_idx] 
                kf_iou_list = keyframe_ious[global_idx]

                kf_rows = []
                for t, k_idx in enumerate(kf_indices):
                    if k_idx is None: continue
                    safe_k_idx = min(max(0, k_idx), len(paths) - 1)
                    
                    try:
                        raw_pil = Image.open(paths[safe_k_idx]).convert('RGB')
                        raw_np = np.array(raw_pil)
                        H, W = raw_np.shape[:2]
                        font = _get_font(H, W, 0.05) 

                        pred_bin = None
                        if t < len(kf_masks_struct):
                            m = kf_masks_struct[t] 
                            if m is not None:
                                if m.dim() == 3: m = m.max(0)[0] 
                                pred_bin = m
                        
                        gt_bin = None
                        if gt_mask_full_raw is not None and img_size is not None:
                            try:
                                gt_slice = self._decode_single_frame(gt_mask_full_raw, safe_k_idx, img_size)
                                if isinstance(gt_slice, np.ndarray) and gt_slice.ndim == 3:
                                    gt_bin = np.max(gt_slice, axis=0)
                                else:
                                    gt_bin = gt_slice
                            except: pass

                        iou_val = kf_iou_list[t] if t < len(kf_iou_list) else 0.0
                        if isinstance(iou_val, list): iou_val = iou_val[0]

                        col_raw = raw_np.copy()
                        col_gt = _apply_overlay(raw_np.copy(), gt_bin, [0, 255, 0])
                        col_pred = _apply_overlay(raw_np.copy(), pred_bin, [255, 0, 0])
                        
                        row_img = np.concatenate([col_raw, col_gt, col_pred], axis=1)
                        
                        row_pil = Image.fromarray(row_img)
                        d = ImageDraw.Draw(row_pil)
                        margin = int(W * 0.02)
                        
                        d.text((margin, margin), f"R{t} | F{safe_k_idx}", font=font, fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0))
                        d.text((margin + W, margin), "GT", font=font, fill=(0,255,0), stroke_width=2, stroke_fill=(0,0,0))
                        d.text((margin + 2*W, margin), f"IoU: {iou_val:.3f}", font=font, fill=(255,50,50), stroke_width=2, stroke_fill=(0,0,0))
                        
                        kf_rows.append(np.array(row_pil))
                    except Exception as e:
                        print(f"KF Vis Error {global_idx}-{t}: {e}")

                if kf_rows:
                    full_vis = np.concatenate(kf_rows, axis=0)
                    Image.fromarray(full_vis).save(os.path.join(save_dir, f"gen{global_idx}_keyframe.jpg"))

                # =========================================================
                # Part B: Temporal Mask Visualization (Original Logic Preserved)
                # =========================================================
                temp_paths = item.get('image_path', [])
                temp_indices = item.get('image_index', [])
                
                if temp_paths:
                    try:
                        loaded_imgs = [np.array(Image.open(p).convert('RGB')) for p in temp_paths]
                        if loaded_imgs:
                            H_t, W_t = loaded_imgs[0].shape[:2]
                            font_t = _get_font(H_t, W_t, 0.10)
                            margin_t = int(W_t * 0.05)
                            
                            temp_rows = []
                            
                            gt_masks_lowres = item.get('mask')
                            gt_row_imgs = []
                            for f_idx, img in enumerate(loaded_imgs):
                                curr = img.copy()
                                fid = temp_indices[f_idx] if f_idx < len(temp_indices) else f_idx
                                
                                if gt_masks_lowres is not None and f_idx < len(gt_masks_lowres):
                                    curr = _apply_overlay(curr, gt_masks_lowres[f_idx], [0, 255, 0])
                                
                                pil_t = Image.fromarray(curr)
                                d = ImageDraw.Draw(pil_t)
                                d.text((margin_t, margin_t), f"{fid}", font=font_t, fill=(255,255,255), stroke_width=1, stroke_fill=(0,0,0))
                                if f_idx == 0: 
                                    d.text((margin_t, H_t - margin_t*4), "GT", font=font_t, fill=(0,255,0), stroke_width=1, stroke_fill=(0,0,0))
                                gt_row_imgs.append(np.array(pil_t))
                            temp_rows.append(np.concatenate(gt_row_imgs, axis=1))
                            
                            flat_masks = low_res_masks_list[global_idx]
                            counts = out_num_objects[global_idx]
                            sample_temp_ious = segmentation_ious[global_idx] if global_idx < len(segmentation_ious) else []

                            cursor = 0
                            for t, count in enumerate(counts):
                                curr_turn_masks = flat_masks[cursor : cursor + count]
                                cursor += count
                                if count == 0: continue
                                
                                valid = [m for m in curr_turn_masks if m is not None]
                                if not valid: continue
                                
                                merged = torch.stack(valid).max(0)[0]
                                round_frame_ious = sample_temp_ious[t] if t < len(sample_temp_ious) else []
                                
                                pred_row_imgs = []
                                for f_idx, img in enumerate(loaded_imgs):
                                    curr = img.copy()
                                    if f_idx < merged.shape[0]:
                                        curr = _apply_overlay(curr, merged[f_idx], [255, 0, 0])
                                    
                                    pil_t = Image.fromarray(curr)
                                    d = ImageDraw.Draw(pil_t)
                                    
                                    if f_idx == 0: 
                                        d.text((margin_t, H_t - margin_t*4), f"R{t}", font=font_t, fill=(255,50,50), stroke_width=1, stroke_fill=(0,0,0))
                                    
                                    if f_idx < len(round_frame_ious):
                                        f_iou = round_frame_ious[f_idx]
                                        iou_txt = f"{f_iou:.2f}"
                                        d.text((W_t - margin_t*5, margin_t), iou_txt, font=font_t, fill=(255,255,0), stroke_width=1, stroke_fill=(0,0,0))

                                    pred_row_imgs.append(np.array(pil_t))
                                temp_rows.append(np.concatenate(pred_row_imgs, axis=1))
                            
                            if len(temp_rows) > 1:
                                full_temp = np.concatenate(temp_rows, axis=0)
                                Image.fromarray(full_temp).save(os.path.join(save_dir, f"gen{global_idx}_temporal.jpg"))

                    except Exception as e:
                        print(f"Temp Vis Error {global_idx}: {e}")
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()
