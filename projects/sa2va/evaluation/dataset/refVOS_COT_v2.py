import os
import json
import copy
import numpy as np
import mmengine
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

# 假设这是你的基类，如果找不到可以替换为 torch.utils.data.Dataset
from .base_eval_dataset import BaseEvalDataset 

SEG_PROMPT_QUESTION = "The query is \"{exp}\". Show me your thought process for finding and segmenting this target. This video has {total_frames} frames."

class DirectResize:
    """SAM2 需要的强制 Resize"""
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Expects a numpy array with shape HxWxC in uint8 format."""
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class RefVOSDatasetCOT_v2(BaseEvalDataset):
    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file,
    ):
        super().__init__()
        self.image_folder = image_folder
        vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
        self.vid2metaid = vid2metaid
        self.videos = list(self.vid2metaid.keys())
        self.mask_dict = mask_dict
        self.text_data = metas

        # --- [优化] 初始化预处理工具 ---
        self.sam2_processor = DirectResize(target_length=1024)
        
        # Qwen 采样参数 (需与模型配置保持一致)
        self.max_video_sample = 20
        self.max_select_K = 8
        self.min_pixel_temporal = 4 * 28 * 28
        self.max_pixel_temporal = 32 * 28 * 28
        self.min_pixel_spatial = 4 * 28 * 28
        self.max_pixel_spatial = 256 * 28 * 28

    def __len__(self):
        return len(self.text_data)

    def _resize_image_to_limit(self, image, min_pixels, max_pixels):
        """辅助函数：按像素限制等比缩放"""
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

    def json_file_preprocess(self, expression_file, mask_file):
        # [修改 2] 判断是否为 GroundMore 数据集
        is_groundmore = 'groundmore' in expression_file.lower()

        with open(expression_file, 'r') as f:
            raw_data = json.load(f)
            # 兼容：有些 JSON 直接是 list 或 dict，GroundMore 包在 "videos" 下
            expression_datas = raw_data['videos'] if 'videos' in raw_data else raw_data

        metas = []
        vid2metaid = {}

        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]
            
            # --- 分支逻辑：获取 Frames ---
            if is_groundmore:
                # GroundMore: 需要去文件夹扫描图片
                # 路径格式: data/.../annotations/<vid_name>/images/frame_xxxx.jpg
                image_dir = os.path.join(self.image_folder, vid_name, 'images')
                if os.path.exists(image_dir):
                    # 获取所有 .jpg / .png 并排序
                    frames_files = sorted([
                        f for f in os.listdir(image_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ])
                else:
                    # 容错：如果找不到目录，给个空列表或打印警告
                    print(f"Warning: Image directory not found for {vid_name} at {image_dir}")
                    frames_files = []
                
                # GroundMore 的 questions 字段解析
                # 结构: "questions": { "1": { "question": ... }, ... }
                questions_dict = vid_express_data.get('questions', {})
                exp_id_list = sorted(list(questions_dict.keys()))
                
                vid_len = len(frames_files)
                
                for exp_id in exp_id_list:
                    q_data = questions_dict[exp_id]
                    meta = {
                        'video': vid_name,
                        'exp': q_data['question'], # 这里字段叫 question
                        'frames': frames_files,    # 包含后缀的文件名列表
                        'exp_id': exp_id,
                        'length': vid_len,
                        'is_groundmore': True      # 标记位
                    }
                    metas.append(meta)
                    if vid_name not in vid2metaid:
                        vid2metaid[vid_name] = []
                    vid2metaid[vid_name].append(len(metas) - 1)

            else:
                # 原有的处理逻辑
                vid_frames = sorted(vid_express_data['frames'])
                vid_len = len(vid_frames)
                exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
                for exp_id in exp_id_list:
                    exp_dict = vid_express_data['expressions'][exp_id]
                    meta = {
                        'video': vid_name,
                        'exp': exp_dict['exp'],
                        'frames': vid_frames,      # 无后缀的文件名列表
                        'exp_id': exp_id,
                        'length': vid_len,
                        'is_groundmore': False
                    }
                    metas.append(meta)
                    if vid_name not in vid2metaid:
                        vid2metaid[vid_name] = []
                    vid2metaid[vid_name].append(len(metas) - 1)

        if mask_file is not None:
            mask_dict = mmengine.load(mask_file)
        else:
            mask_dict = None
        return vid2metaid, metas, mask_dict

    def __getitem__(self, index):
        video_obj_info = copy.deepcopy(self.text_data[index])
        exp = video_obj_info['exp']
        video_id = video_obj_info['video']
        raw_frames_list = video_obj_info['frames']
        is_groundmore = video_obj_info.get('is_groundmore', False)

        data_dict = {}

        if is_groundmore:
            # GroundMore: video_id/images/frame_xxx.jpg (frames 列表里已有后缀)
            frames_files = [
                os.path.join(self.image_folder, video_id, 'images', frame_file) 
                for frame_file in raw_frames_list
            ]
        else:
            # 原始逻辑: video_id/frame_xxx.jpg (frames 列表里无后缀)
            frames_files = [
                os.path.join(self.image_folder, video_id, frame_file + ".jpg") 
                for frame_file in raw_frames_list
            ]

        
        # 1. 加载原始图片
        images = []
        ori_width, ori_height = None, None
        for frame_path in frames_files:
            try:
                frame_image = Image.open(frame_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {frame_path}: {e}")
                # 简单的容错处理，生成全黑图
                frame_image = Image.new('RGB', (256, 256))
            
            if ori_height is None:
                ori_width, ori_height = frame_image.size
            images.append(frame_image)

        # 2. 计算采样索引
        vid_len = len(images)
        if vid_len > self.max_video_sample:
            video_sample_index = np.linspace(0, vid_len-1, self.max_video_sample, dtype=int)
        else:
            video_sample_index = np.arange(vid_len)
        
        if vid_len > self.max_select_K:
            image_sample_index = np.linspace(0, vid_len-1, self.max_select_K, dtype=int)
        else:
            image_sample_index = np.arange(vid_len)

        # --- [优化] CPU 并行预处理核心区域 ---
        
        # A. SAM2 Resize (最耗时步骤)
        # 转换为 numpy array 存储，避免 PyTorch DataLoader 在不同 worker 间传递 Tensor/PIL 的开销问题
        processed_sam2_images = []
        for img in images:
            img_np = np.array(img) 
            processed_sam2_images.append(self.sam2_processor.apply_image(img_np))
        
        # B. Qwen Initial Frames Resize
        qwen_video_frames = []
        qwen_spatial_frames = []

        for idx in video_sample_index:
            qwen_video_frames.append(self._resize_image_to_limit(
                images[idx], self.min_pixel_temporal, self.max_pixel_temporal
            ))
            
        for idx in image_sample_index:
            qwen_spatial_frames.append(self._resize_image_to_limit(
                images[idx], self.min_pixel_spatial, self.max_pixel_spatial
            ))
        # ------------------------------------

        data_dict['type'] = 'video'
        data_dict['index'] = index
        data_dict['video_id'] = video_id
        
        # 原始 images 仍需保留，因为 CoT 可能会动态 select 初始采样之外的帧
        data_dict['images'] = images 
        
        # 预处理后的数据
        data_dict['processed_sam2_images'] = processed_sam2_images # List[np.ndarray] (1024, 1024, 3)
        data_dict['qwen_video_frames'] = qwen_video_frames         # List[PIL.Image]
        data_dict['qwen_spatial_frames'] = qwen_spatial_frames     # List[PIL.Image]
        data_dict['video_sample_index'] = video_sample_index
        data_dict['image_sample_index'] = image_sample_index

        data_dict['exp_id'] = video_obj_info['exp_id']
        data_dict['frames'] = video_obj_info['frames']
        data_dict['text_prompt'] = SEG_PROMPT_QUESTION.format(exp=exp, total_frames=len(frames_files))
        data_dict['image_folder'] = self.image_folder
        data_dict['length'] = video_obj_info['length']
        data_dict['ori_height'] = ori_height
        data_dict['ori_width'] = ori_width

        return data_dict