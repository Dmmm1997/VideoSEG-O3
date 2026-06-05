import os
from typing import Literal
import pickle
from typing import Literal, Optional, Dict, List, Any

import torch
import numpy as np
import copy
import json
import random
import pycocotools.mask as maskUtils

from projects.sa2va.datasets.common import SEG_QUESTIONS, ANSWER_LIST
from projects.sa2va.datasets.base import Sa2VABaseDataset
from projects.sa2va.datasets.data_utils import sam2_path_patch, get_video_frames, decode_masklet, opencvimg_to_pil
import re

def extract_select_dict(text: str) -> Optional[Dict[str, Any]]:
    """
    从包含 <select> 标签的字符串中提取并解析内部的 JSON 字典。
    """
    # 非贪婪匹配模式，用于捕获 <select>...</select> 之间的内容
    regex_pattern = r'<select>(.*?)</select>'
    
    match = re.search(regex_pattern, text)
    
    if match:
        json_str = match.group(1)
        try:
            # 使用 json.loads 安全地将 JSON 字符串转换为 Python 字典
            result_dict = json.loads(json_str)
            return result_dict
        except json.JSONDecodeError:
            # 如果解析失败，返回 None
            return None
    
    # 如果没有找到匹配的 <select> 标签，返回 None
    return None

class COTRefVOS(Sa2VABaseDataset):

    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 select_number=5,
                 sampled_frames=5,
                 sam_sampled_frames=5,
                 dataset_type: Literal['default']='default',
                 **kwargs):
        
        # Initialize base class
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
        
        # RefVOS-specific configurations
        self.dataset_type = dataset_type
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        assert expression_file and tokenizer
        # Adjusted pixel ranges to match Qwen3VL expectations
        self.min_pixel_temporal = 32*28*28  # Increased minimum
        self.max_pixel_temporal = 64*28*28  # Increased maximum

        self.min_pixel_spatial = 128*28*28   # Reduced minimum
        self.max_pixel_spatial = 256*28*28  # Reduced maximum

        self.sam_sampled_frames = sam_sampled_frames

        assert mask_file is not None
        vid2metaid, mask_dict = self.json_file_preprocess(expression_file, mask_file)
        self.video_infos = vid2metaid
        # self.videos = list(self.video_infos.keys())
        self.mask_dict = mask_dict

        self.image_folder = image_folder
        self.max_temporal_frames_perround = 10 # 每一轮新增的最大temporal frames的数量

        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.TEMP_CONTEXT_TOKEN = '<|temporal_pad|>'
            self.SPATIAL_CONTEXT_TOKEN = '<|spatial_pad|>'

    def _process_multiple_images(self, images, min_pixels, max_pixels):
        """
        Process multiple images (for video datasets) and return pixel values and number of tokens.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dictionary containing processed image data
        """
        result = {}
        pixel_values = []
        extra_pixel_values = []
        
        # Process each image
        for image in images:
            image = image.convert('RGB')
            ori_width, ori_height = image.size
            
            # Process for grounding if needed
            if hasattr(self, 'extra_image_processor') and self.extra_image_processor is not None:
                g_image = np.array(image)
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(g_pixel_values)

            if self.preprocessor is not None:
                # Store images for batch processing
                pixel_values.append(image)
            else:
                # Apply transforms immediately
                transformed = self.transformer(image)
                pixel_values.append(transformed)

        # Process images based on preprocessor availability
        if self.preprocessor is not None:
            if self.arch_type == 'qwen':
                merge_length = self.preprocessor.image_processor.merge_size ** 2
                _data_dict = self.preprocessor.image_processor(
                    images=images, min_pixels=min_pixels, max_pixels=max_pixels
                )
                num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() // merge_length)
                num_frames = _data_dict['image_grid_thw'].shape[0]
                num_total_tokens = num_frame_tokens * num_frames
                result.update(_data_dict)
                result['num_frame_tokens'] = num_frame_tokens
                result['num_frames'] = num_frames
            elif self.arch_type == 'llava':
                raise NotImplementedError("LLaVA preprocessor not implemented for multiple image mode")
            else:
                raise NotImplementedError(f"Preprocessor not implemented for {self.arch_type}")
        else:
            pixel_values = torch.stack(pixel_values, dim=0)  # (n_f, 3, h, w)
            result['pixel_values'] = pixel_values
            num_total_tokens = len(images) * self.patch_token

        if extra_pixel_values:
            result['g_pixel_values'] = extra_pixel_values

        result['num_image_tokens'] = num_total_tokens
        return result

    def real_len(self):
        return len(self.video_infos)

    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)

        if mask_file.endswith('.pkl'):
            with open(mask_file, 'rb') as f:
                mask_dict = pickle.load(f)
        elif mask_file.endswith('.json'):
            with open(mask_file, 'r') as f:
                mask_dict = json.load(f)
        else:
            raise NotImplementedError

        return expression_datas, mask_dict

    def decode_mask(self, video_masks, image_size):
        ret_masks = []
        for object_masks in video_masks:
            # None object
            if len(object_masks) == 0:
                if len(ret_masks) != 0:
                    _object_masks = ret_masks[0] * 0
                else:
                    _object_masks = np.zeros(
                        (self.sampled_frames, image_size[0], image_size[1]), dtype=np.uint8)
            else:
                _object_masks = []
                for i_frame in range(len(object_masks[0])):
                    _mask = np.zeros(image_size, dtype=np.uint8)
                    for i_anno in range(len(object_masks)):
                        if object_masks[i_anno][i_frame] is None:
                            continue
                        m = maskUtils.decode(object_masks[i_anno][i_frame])
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        _mask = _mask | m
                    _object_masks.append(_mask)
                _object_masks = np.stack(_object_masks, axis=0)
            ret_masks.append(_object_masks)
        _shape = ret_masks[0].shape
        for item in ret_masks:
            if item.shape != _shape:
                print([_ret_mask.shape for _ret_mask in ret_masks])
                return None
        ret_masks = np.stack(ret_masks, axis=0)  # (n_obj, n_frames, h, w)
        ret_masks = torch.from_numpy(ret_masks)
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks

    def prepare_data(self, index):
        """Prepare data for a given index using unified base class methods."""
        index = index % self.real_len()
        selected_video_objects = self.video_infos[index]
        # if selected_video_objects['round_num']>1:
        #     print(10)
        
        data_dict = self.dataset_map_fn(selected_video_objects, select_k=self.sampled_frames)
        
        if data_dict is None:
            return None

        out_data_dict = {}
        
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('images', None) is not None:
            try:
            # Load images from paths
                images = []
                for img_path in data_dict['images']:
                    full_img_path = os.path.join(self.image_folder, img_path)
                    image = self._read_image(full_img_path)
                    if image is None:
                        raise Exception(f"Failed to read image: {full_img_path}")
                    images.append(image)

                # Process multiple images using base class method
                image_data = self._process_multiple_images(images, min_pixels=self.min_pixel_temporal, max_pixels=self.max_pixel_temporal)
                # select SAM image

                num_frames = len(images)
                if self.sam_sampled_frames<num_frames:
                    index = np.random.choice(num_frames, self.sam_sampled_frames, replace=False)
                else:
                    index = np.arange(0, num_frames)
                index.sort()
                image_data["g_pixel_values"] = [image_data["g_pixel_values"][i] for i in index]
                out_data_dict["masks"] = out_data_dict["masks"][index]
                image_token_str = self._create_token_string(image_data['num_image_tokens'], num_frames, self.IMG_CONTEXT_TOKEN)

                image_data_total = copy.deepcopy(image_data)
                image_data_total["temporal_frame_tokens"] = 0
                image_data_total["spatial_frame_tokens"] = 0
                
                image_token_str_list = [image_token_str]
                for round in data_dict["multi_round_frames"]:
                    temporal_frames, key_frame, temporal_frames_indices, key_frame_indice = round
                    # read temporal frames
                    temporal_frames = [self._read_image(os.path.join(self.image_folder, img_path)) for img_path in temporal_frames]
                    temporal_frames_data = self._process_multiple_images(temporal_frames, min_pixels=self.min_pixel_temporal, max_pixels=self.max_pixel_temporal)
                    temporal_token_str = self._create_token_string(temporal_frames_data['num_image_tokens'], len(temporal_frames), self.TEMP_CONTEXT_TOKEN)

                    # read key frame
                    key_frame = self._read_image(os.path.join(self.image_folder, key_frame))
                    key_frame_data = self._process_multiple_images([key_frame], min_pixels=self.min_pixel_spatial, max_pixels=self.max_pixel_spatial)
                    key_token_str = self._create_token_string(key_frame_data['num_image_tokens'], 1, self.SPATIAL_CONTEXT_TOKEN)

                    image_data_total['image_grid_thw'] = torch.cat((image_data_total['image_grid_thw'], temporal_frames_data['image_grid_thw']), dim=0)
                    image_data_total['image_grid_thw'] = torch.cat((image_data_total['image_grid_thw'], key_frame_data['image_grid_thw']), dim=0)
                    image_data_total['pixel_values'] = torch.cat((image_data_total['pixel_values'], temporal_frames_data['pixel_values']), dim=0)
                    image_data_total['pixel_values'] = torch.cat((image_data_total['pixel_values'], key_frame_data['pixel_values']), dim=0)
                    image_data_total["temporal_frame_tokens"] = temporal_frames_data["num_frame_tokens"]
                    image_data_total["spatial_frame_tokens"] = key_frame_data["num_frame_tokens"]
                    image_data_total["num_image_tokens"] += key_frame_data["num_image_tokens"]
                    image_data_total["num_image_tokens"] += temporal_frames_data["num_image_tokens"]

                    image_token_str_list.append([temporal_token_str, key_token_str, temporal_frames_indices, key_frame_indice])

                out_data_dict.update(image_data_total)
                # Create video token string
                
                # Process conversations using unified method
                conversations = self._process_conversations_for_encoding(
                    data_dict['conversations'], image_token_str_list, is_video=True
                )
                
                # Handle token expansion for qwen if needed
                if self.arch_type == 'qwen' and 'num_frame_tokens' in image_data_total:
                    conversations = self._expand_video_tokens(
                        conversations, image_data_total['num_frame_tokens'], image_data_total["temporal_frame_tokens"], image_data_total["spatial_frame_tokens"], image_data_total['num_image_tokens']
                    )
                
                # Get input/labels using base class method
                token_dict = self.get_inputid_labels(conversations)
                out_data_dict.update(token_dict)
                
            except Exception as e:
                print(f'Error processing images: {e}', flush=True)
                return None
        else:
            # No images case
            conversations = self._process_conversations_for_encoding(data_dict['conversations'], None, is_video=True)
            token_dict = self.get_inputid_labels(conversations)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            out_data_dict['masks'] = None

        out_data_dict['type'] = 'video'
        return out_data_dict
    
    def dataset_map_fn(self, data_dict, select_k=5, temporal_max_frames=3):
        images = []

        len_frames = len(data_dict['frames'])

        # prepare images, random select k frames
        if len_frames > select_k + 1:
            # selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
            selected_frame_indexes = np.linspace(0, len_frames - 1, select_k, dtype=int)
        else:
            selected_frame_indexes = np.arange(len_frames)
            # selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
        selected_frame_indexes.sort()
        
        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict['frames'][selected_frame_index]
            images.append(os.path.join(data_dict['video_name'], frame_id + '.jpg'))

        # prepare masks
        # one exp can have multiple annos
        video_masks = []
        anno_ids = data_dict['anno_id']
        obj_masks = []
        for anno_id in anno_ids:
            anno_id = str(anno_id)
            frames_masks = self.mask_dict[anno_id]
            frames_masks_ = []
            for frame_idx in selected_frame_indexes:
                frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
            obj_masks.append(frames_masks_)
        video_masks.append(obj_masks)

        # prepare conversation
        multi_round_frames = []
        for round_id in range(len(data_dict["cot_trace"])):
            round = data_dict["cot_trace"][round_id]
            matched_times = extract_select_dict(data_dict["conversations"][(round_id+1)*2]["value"])
            if not matched_times:
                break
            assert matched_times["start"] == round["start_frame"]
            assert matched_times["end"] == round["end_frame"]
            assert matched_times["keyframe"] == round["key_frame"]
            start_frame = round["start_frame"]
            end_frame = round["end_frame"]
            key_frame_indice = round["key_frame"]
            if end_frame-start_frame>self.max_temporal_frames_perround-1:
                this_round_sampled_indexes = np.linspace(start_frame, end_frame, self.max_temporal_frames_perround, dtype=int)
            else:
                this_round_sampled_indexes = np.arange(start_frame, end_frame)

            temporal_frames = []
            # temporal frames per round
            for selected_frame_index in this_round_sampled_indexes:
                frame_id = data_dict['frames'][selected_frame_index]
                temporal_frames.append(os.path.join(data_dict['video_name'], frame_id + '.jpg'))
            # key frame this round
            key_frame = os.path.join(data_dict['video_name'], data_dict['frames'][key_frame_indice] + '.jpg')
            multi_round_frames.append([temporal_frames, key_frame, this_round_sampled_indexes, key_frame_indice])
        

        # read image size from the first image
        first_image_path = images[0]
        first_image_path = os.path.join(self.image_folder, first_image_path)
        first_image = self._read_image(first_image_path)
        if first_image is None:
            return None
        
        # switch height and width (PIL system (WH vs HW system)
        _image_size = first_image.size
        image_size = (_image_size[1], _image_size[0])
        masks = self.decode_mask(video_masks, image_size=image_size)
        if masks is None:
            return None

        ret = {'images': images, 'conversations': data_dict["conversations"], 'masks': masks, 'multi_round_frames': multi_round_frames}
        return ret

    @property
    def modality_length(self):
        return [self._get_modality_length_default(30000) for _ in range(self.real_len())]

    def mock_prepare_data(self, index):
        """
        Mock version of prepare_data that only checks image existence.
        Useful for testing and validation without loading full data.
        
        Returns:
            dict with status information or None if images don't exist
        """
        if self.dataset_type in ['refsav']:
            mock_data_dict = {}
            video_info = self.video_infos[self.videos[index]]
            video_path = os.path.join(self.image_folder, video_info['video_path'])
            anno_path = os.path.join(self.image_folder, video_info['anno_path'])
            video_path, anno_path = sam2_path_patch(video_path, anno_path)

            if not os.path.exists(video_path):
                print(f'Video path does not exist: {video_path}', flush=True)
                return None
            if not os.path.exists(anno_path):
                print(f'Annotation path does not exist: {anno_path}', flush=True)
                return None
            
            mock_data_dict.update({
                'video_name': video_info['video_path'],
                'has_images': True,
                'num_frames': 5,
                'num_objects': len(video_info['objects']),
                'status': 'valid',
                'type': 'video'
            })
            return mock_data_dict

        index = index % self.real_len()
        selected_video_objects = self.video_infos[self.videos[index]]
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
        else:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        mock_data_dict = {}
        
        # Check image existence
        len_frames = len(video_objects_infos[0]['frames'])
        if len_frames > self.sampled_frames + 1:
            selected_frame_indexes = np.random.choice(len_frames, self.sampled_frames, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, self.sampled_frames, replace=True)
        selected_frame_indexes.sort()
        
        # Check if images exist
        for selected_frame_index in selected_frame_indexes:
            frame_id = video_objects_infos[0]['frames'][selected_frame_index]
            image_path = os.path.join(video_objects_infos[0]['video'], frame_id + '.jpg')
            full_image_path = os.path.join(self.image_folder, image_path)
            if not self._check_image_exists(full_image_path):
                print(f'Image does not exist: {full_image_path}', flush=True)
                return None
        
        mock_data_dict.update({
            'video_name': video_objects_infos[0]['video'],
            'has_images': True,
            'num_frames': len(selected_frame_indexes),
            'num_objects': len(video_objects_infos),
            'num_conversations': len(video_objects_infos) * 2,  # Each object creates 2 conversation turns
            'status': 'valid',
            'type': 'video'
        })
            
        return mock_data_dict
    
    def _process_conversations_for_encoding(self, conversations: List[Dict], image_token_str: Optional[str] = None, 
                                          is_video: bool = False) -> List[Dict]:
        """
        Process conversations to prepare for tokenization.
        
        Args:
            conversations: List of conversation messages
            image_token_str: Image token string to replace <image> placeholders
            is_video: Whether this is video data (affects token placement)
            
        Returns:
            List of processed conversation turns
        """
        # Handle different input formats
        if conversations and 'input' in conversations[0] and 'output' in conversations[0]:
            # Already in the correct format (from video datasets)
            return conversations
            
        input_text = ''
        out_conversation = []
        
        # Skip leading GPT messages
        while conversations and conversations[0]['from'] == 'gpt':
            conversations = conversations[1:]
        
        conv_idx = 0
        
        conversation = defaultdict(str)
        for msg in conversations:
            if msg['from'] == "system":
                value = msg['value']
                conversation["system"] = value
            elif msg['from'] == 'human':
                value = msg['value']
                # Handle image token replacement
                if conv_idx==0:
                    value = image_token_str[0][0] + f"\nHere are {len(image_token_str[0][2])} low-resolution frames in this video (frame indice is {str(image_token_str[0][2])})\n" + image_token_str[0][1] + f"Here are {len(image_token_str[0][3])} high-resolution frames in this video (frame indice:{str(image_token_str[0][3])}).\n" + value
                    value = value.strip()
                else:
                    new_value = f"\nHere are {len(image_token_str[conv_idx][2])} low-resolution frames in this video (frame indices are {str(image_token_str[conv_idx][2])})\n" + image_token_str[conv_idx][1] + f"Here is key frames(frame indice:{str(image_token_str[conv_idx][3])}).\n" + value
                    value = value.replace('<image>', new_value)
                conv_idx+=1
                # input_text += value
                conversation["input"] += value
            elif msg['from'] == 'gpt':
                conversation["output"] += msg['value'].strip()
                out_conversation.append(conversation)
                conversation = defaultdict(str)
                # input_text = ''

            else:
                raise NotImplementedError(f"Unknown message role: {msg['from']}")
                    
        return out_conversation

    def _expand_video_tokens(self, conversations: List[Dict], num_frame_tokens: int, num_temporal_tokens:int, num_spatial_tokens:int, num_total_tokens: int) -> List[Dict]:
        """
        Expand video tokens for architectures that need post-processing (like qwen).
        
        Args:
            conversations: Processed conversations
            num_frame_tokens: Tokens per frame
            num_total_tokens: Total video tokens
            
        Returns:
            Updated conversations with expanded tokens
        """
        if conversations and self.arch_type == 'qwen' and hasattr(self, 'patch_token') and self.patch_token == 1:
            # For qwen, expand the single tokens to frame tokens
            total_image_count = 0
            for ind, conv in enumerate(conversations):
                input_str = conv['input']
                input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
                input_str = input_str.replace(self.TEMP_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_temporal_tokens)
                input_str = input_str.replace(self.SPATIAL_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_spatial_tokens)
                total_image_count += input_str.count(self.IMG_CONTEXT_TOKEN)
                conversations[ind]['input'] = input_str
            assert total_image_count == num_total_tokens, \
                f"Token count mismatch: expected {num_total_tokens}, got {total_image_count}"
        return conversations

    def _create_token_string(self, num_tokens: int, num_frames: int = 1, context_token: str = '|<image_pad>|') -> str:
        """
        Create token string for images or videos.
        
        Args:
            num_tokens: Total number of tokens
            num_frames: Number of frames (1 for image, >1 for video)
            
        Returns:
            Token string with proper formatting
        """
        # Video case - create frame tokens
        if self.arch_type == 'qwen' and hasattr(self, 'patch_token') and self.patch_token == 1:
            # For qwen with patch_token=1, we use single tokens that will be expanded later
            frame_token_str = f'{self.IMG_START_TOKEN}{context_token}{self.IMG_END_TOKEN}'
        else:
            # For other cases, use tokens per frame
            tokens_per_frame = num_tokens // num_frames
            frame_token_str = f'{self.IMG_START_TOKEN}{context_token * tokens_per_frame}{self.IMG_END_TOKEN}'
        
        # Repeat for all frames with newlines
        frame_tokens = (frame_token_str + '\n') * num_frames
        return frame_tokens.strip()

if __name__ == '__main__':
    from transformers import AutoTokenizer, Qwen2_5_VLProcessor, Qwen3VLProcessor
    from xtuner.utils import PROMPT_TEMPLATE
    from projects.sa2va.models import DirectResize
    COTRefVOS_dataset = COTRefVOS(
        image_folder="/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/Sa2VA/data/video_datas/revos",
        expression_file="/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/DataConstruction/datapipeline/revos/generation_data/qwen3vl235bv2/5-COT_final_with_images.json",
        mask_file='/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/Sa2VA/data/video_datas/revos/mask_dict.json',
        repeats=1,
        select_number=1,
        sampled_frames=30,
        dataset_type='default',
        arch_type='qwen',
        preprocessor=dict(
            type=Qwen3VLProcessor.from_pretrained, # or Qwen3VLProcessor for Qwen3VL-based models
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
    from tqdm import tqdm
    for i in tqdm(range(len(COTRefVOS_dataset))):
        try:
            COTRefVOS_dataset.prepare_data(i)
        except:
            print(f"Error processing item {i}")
    