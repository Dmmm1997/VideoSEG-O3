import copy
import json
import random
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from projects.open_r1.constants import system_prompt_registry, question_template_registry, answer_template_registry
from qwen_vl_utils import smart_resize
from torchvision.transforms.functional import resize as resize_api
import numpy as np
from pycocotools import mask as maskUtils
import glob
import cv2

local_rank = int(os.environ.get("LOCAL_RANK", -1))

def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence

def resize_longest(image: Image.Image, longest_side_length=640):
    """
    Resize the image so that its longest side is scaled to the specified length,
    while maintaining the aspect ratio.

    :param image: The PIL.Image object to resize.
    :param longest_side_length: The length of the longest side after resizing.
    :return: The resized PIL.Image object.
    """
    # Get the original width and height of the image
    original_width, original_height = image.size

    # Determine which side is the longest
    if original_width > original_height:
        scale_factor = longest_side_length / original_width
    else:
        scale_factor = longest_side_length / original_height

    # Calculate the new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image

def resize_shortest(image: Image.Image, shortest_side_length=640):
    """
    Resize the image so that its shortest side is scaled to the specified length,
    while maintaining the aspect ratio.

    :param image: The PIL.Image object to resize.
    :param shortest_side_length: The length of the shortest side after resizing.
    :return: The resized PIL.Image object.
    """
    original_width, original_height = image.size

    # Determine which side is the shortest
    if original_width < original_height:
        scale_factor = shortest_side_length / original_width
    else:
        scale_factor = shortest_side_length / original_height

    # Calculate the new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image

def mask_to_bounding_box(mask):
    """
    Given a mask, compute the bounding box of the foreground.

    :param mask: 2D ndarray containing values 0, 1, 255
                 1: indicates foreground
                 0: indicates background
                 255: indicates ignore
    :return: The bounding box coordinates as a JSON-style list.
    """
    # Find the coordinates of the foreground pixels (value == 1)
    foreground_positions = np.argwhere(mask == 1)

    # If no foreground exists, return an empty box
    if foreground_positions.size == 0:
        return [0, 0, 0, 0]

    # Compute the minimum and maximum boundaries of the bounding box
    ymin, xmin = foreground_positions.min(axis=0)
    ymax, xmax = foreground_positions.max(axis=0)

    # Prepare the bounding box data
    bounding_box = [
        int(xmin),
        int(ymin),
        int(xmax),
        int(ymax),
    ]

    return bounding_box

class ReasonSegDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        base_image_dir = "./datasets",
        solution_format="str",
    ):
        self.script_args = script_args
        self.solution_format = solution_format
        self.images = glob.glob(os.path.join(base_image_dir, "reason_seg", "ReasonSeg", "train", "*.jpg"))
        self.images = sorted(self.images)

        self.question_template = question_template_registry[script_args.question_template]
        self.system_template = system_prompt_registry["default"]
        self.coord_norm_type = script_args.coord_norm_type
        print(f'coord_norm_type = {self.coord_norm_type}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        json_path = image_path.replace(".jpg", ".json")


        image = Image.open(image_path).convert(mode="RGB")
        origin_width, origin_height = image.size
        image = resize_longest(image, longest_side_length=640)
        width, height = image.size
        min_pixels = self.script_args.min_pixels
        max_pixels = self.script_args.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        llm_image = image.resize((resized_width, resized_height))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_image = cv2.resize(image, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(sam_image).permute(2, 0, 1).contiguous()
        sam_image = (sam_image-self.pixel_mean) / self.pixel_std

        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        box = mask_to_bounding_box(mask_json)
        xmin, ymin, xmax, ymax = box
        if self.coord_norm_type=="qwen2vl":
            solution = [
                round(xmin / origin_width * 1000),
                round(ymin / origin_height * 1000),
                round(xmax / origin_width * 1000),
                round(ymax / origin_height * 1000)
            ]
        elif self.coord_norm_type=="qwen2p5vl":
            solution = [
                round(xmin / origin_width * resized_width),
                round(ymin / origin_height * resized_height),
                round(xmax / origin_width * resized_width),
                round(ymax / origin_height * resized_height),
            ]
        else:
            raise NotImplementedError("Unknown coord_norm_type")
        if self.solution_format == "str":
            solution = str(solution)
        elif self.solution_format == "list":
            solution = solution
        else:
            raise ValueError(f"Unknown solution format: {self.solution_format}")


        problem = random.choice(sampled_sents)
        prompt = json.dumps(
            [
                {"role": "system", "content": self.system_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem)}
                ]},
            ]
        )

        masks = [mask_json]
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        masks = (masks==1).float()

        return {
            "image": llm_image,
            "problem": problem,
            "solution": solution,
            "prompt": prompt,
            "sam_image": sam_image,
            "mask": masks[0],
            "image_path": image_path
        }

class ReferSegDataset(torch.utils.data.Dataset):
    '''
    Each item corresponds to a referring mask.
    During __getitem__, one referring sentence is randomly selected from multiple candidates.
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        base_image_dir = "./datasets",
        solution_format="str",
    ):
        super().__init__()
        self.script_args = script_args
        self.base_image_dir = base_image_dir
        self.system_prompt_template = system_prompt_registry[script_args.system_prompt_template]
        self.question_template = question_template_registry[script_args.question_template]
        self.answer_template = answer_template_registry[script_args.answer_template]
        self.refer_seg_ds_list = script_args.refer_seg_ds.split(",")
        self.coord_norm_type = script_args.coord_norm_type
        self.list_data_dict = []
        self.refer_annotations = {}
        self.total_images = 0

        # format solution
        self.solution_format = solution_format

        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            refer_api = REFER(os.path.join(base_image_dir, "refer_seg"), ds, splitBy)

            ref_ids_train = refer_api.getRefIds(split="train")
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            self.list_data_dict.extend(refs_train)

            self.refer_annotations.update(refer_api.Anns)

        if self.script_args.train_sample_size is not None and self.script_args.train_sample_size < len(self.list_data_dict):
            self.list_data_dict = random.sample(self.list_data_dict, self.script_args.train_sample_size)
            print(f"Loaded {len(self.list_data_dict)} samples from {self.base_image_dir}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        ref = self.list_data_dict[idx]
        if "file_name" in ref:
            image_path = f"COCO_train2014_{ref['image_id']:012}.jpg"
            image_path = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2014", image_path)
        else:
            # refclef
            image_id = ref["image_id"]
            image_path = os.path.join(self.base_image_dir, "refer_seg/images/saiapr_tc-12/{:02d}/images/{}.jpg".format(image_id//1000, image_id))

        image = Image.open(image_path).convert(mode="RGB")
        width, height = image.size
        min_pixels = self.script_args.min_pixels
        max_pixels = self.script_args.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        llm_image = image.resize((resized_width, resized_height))

        sam_image = resize_api(image, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).contiguous()
        sam_image = (sam_image-self.pixel_mean) / self.pixel_std

        problem = random.choice(ref["sentences"])["sent"]
        prompt = json.dumps(
            [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem)},
                ]},
            ],
        )
        ann = self.refer_annotations[ref["ann_id"]]
        xmin, ymin, box_w, box_h = ann["bbox"]
        xmax = xmin + box_w
        ymax = ymin + box_h
        if self.coord_norm_type=="qwen2vl":
            box = [
                round(xmin / width * 1000),
                round(ymin / height * 1000),
                round(xmax / width * 1000),
                round(ymax / height * 1000)
            ]
        elif self.coord_norm_type=="qwen2p5vl":
            box = [
                round(xmin / width * resized_width),
                round(ymin / height * resized_height),
                round(xmax / width * resized_width),
                round(ymax / height * resized_height),
            ]
        else:
            raise NotImplementedError

        if self.solution_format == "str":
            box = str(box)
        elif self.solution_format == "list":
            box = box
        else:
            raise ValueError(f"Unknown solution format: {self.solution_format}")

        if len(ann["segmentation"]) == 0:
            # import pdb;pdb.set_trace()
            mask = np.zeros((height, width)).astype(
                np.uint8
            )

        if type(ann["segmentation"][0]) == list:  # polygon
            rle = maskUtils.frPyObjects(
                ann["segmentation"], height, width
            )
        else:
            rle = ann["segmentation"]
            for i in range(len(rle)):
                if not isinstance(rle[i]["counts"], bytes):
                    rle[i]["counts"] = rle[i]["counts"].encode()
        mask = maskUtils.decode(rle)
        mask = np.sum(
            mask, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        mask = mask>0
        mask = mask.astype(np.uint8)  # convert to np.uint8
        mask = torch.from_numpy(mask)
        return {
            "image": llm_image,
            "problem": problem,
            "solution": box,
            "prompt": prompt,
            "sam_image": sam_image,
            "mask": mask,
            "image_path": image_path,
            "ref_meta": ref
        }

class ReferSegDataset_2(torch.utils.data.Dataset):
    '''
    Each item corresponds to a referring mask.
    During __getitem__, one referring sentence is randomly selected from multiple candidates.
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        base_image_dir = "./datasets",
        solution_format="str",
    ):
        super().__init__()
        self.script_args = script_args
        self.base_image_dir = base_image_dir
        self.system_prompt_template = system_prompt_registry[script_args.system_prompt_template]
        self.question_template = question_template_registry[script_args.question_template]
        self.answer_template = answer_template_registry[script_args.answer_template]
        self.refer_seg_ds_list = script_args.refer_seg_ds.split(",")
        self.coord_norm_type = script_args.coord_norm_type
        self.list_data_dict = []
        self.refer_annotations = {}
        self.total_images = 0

        # format solution
        self.solution_format = solution_format

        for ds in self.refer_seg_ds_list:
            if ds == "refcocog":
                splitBy = "umd"
            else:
                splitBy = "unc"

            refer_api = REFER(os.path.join(base_image_dir, "refer_seg"), ds, splitBy)

            ref_ids_train = refer_api.getRefIds(split="train")
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            self.list_data_dict.extend(refs_train)

            self.refer_annotations.update(refer_api.Anns)

        if self.script_args.train_sample_size is not None and self.script_args.train_sample_size < len(self.list_data_dict):
            self.list_data_dict = random.sample(self.list_data_dict, self.script_args.train_sample_size)
            print(f"Loaded {len(self.list_data_dict)} samples from {self.base_image_dir}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        ref = self.list_data_dict[idx]
        if "file_name" in ref:
            image_path = f"COCO_train2014_{ref['image_id']:012}.jpg"
            image_path = os.path.join(self.base_image_dir, "refer_seg/images/mscoco/images/train2014", image_path)
        else:
            # refclef
            image_id = ref["image_id"]
            image_path = os.path.join(self.base_image_dir, "refer_seg/images/saiapr_tc-12/{:02d}/images/{}.jpg".format(image_id//1000, image_id))

        image = Image.open(image_path).convert(mode="RGB")
        width, height = image.size
        min_pixels = self.script_args.min_pixels
        max_pixels = self.script_args.max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        llm_image = image.resize((resized_width, resized_height))

        sam_image = resize_api(image, (self.sam_img_size, self.sam_img_size))
        sam_image = torch.from_numpy(np.array(sam_image)).permute(2, 0, 1).contiguous()
        sam_image = (sam_image-self.pixel_mean) / self.pixel_std

        problem = random.choice(ref["sentences"])["sent"]
        prompt = json.dumps(
            [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.question_template.format(question=problem, total_frames=1)},
                ]},
            ],
        )
        ann = self.refer_annotations[ref["ann_id"]]
        xmin, ymin, box_w, box_h = ann["bbox"]
        xmax = xmin + box_w
        ymax = ymin + box_h
        if self.coord_norm_type=="qwen2vl":
            box = [
                round(xmin / width * 1000),
                round(ymin / height * 1000),
                round(xmax / width * 1000),
                round(ymax / height * 1000)
            ]
        elif self.coord_norm_type=="qwen2p5vl":
            box = [
                round(xmin / width * resized_width),
                round(ymin / height * resized_height),
                round(xmax / width * resized_width),
                round(ymax / height * resized_height),
            ]
        else:
            raise NotImplementedError

        if self.solution_format == "str":
            box = str(box)
        elif self.solution_format == "list":
            box = box
        else:
            raise ValueError(f"Unknown solution format: {self.solution_format}")

        if len(ann["segmentation"]) == 0:
            # import pdb;pdb.set_trace()
            mask = np.zeros((height, width)).astype(
                np.uint8
            )

        if type(ann["segmentation"][0]) == list:  # polygon
            rle = maskUtils.frPyObjects(
                ann["segmentation"], height, width
            )
        else:
            rle = ann["segmentation"]
            for i in range(len(rle)):
                if not isinstance(rle[i]["counts"], bytes):
                    rle[i]["counts"] = rle[i]["counts"].encode()
        mask = maskUtils.decode(rle)
        mask = np.sum(
            mask, axis=2
        )  # sometimes there are multiple binary map (corresponding to multiple segs)
        mask = mask>0
        mask = mask.astype(np.uint8)  # convert to np.uint8
        mask = torch.from_numpy(mask)
        return {
            "image": llm_image,
            "problem": problem,
            "solution": box,
            "prompt": prompt,
            "sam_image": sam_image,
            "mask": mask,
            "image_path": image_path,
            "ref_meta": ref
        }

class ReferVideoSegDataset(torch.utils.data.Dataset):
    '''
    Each item corresponds to a referring mask.
    During __getitem__, one referring sentence is randomly selected from multiple candidates.
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    sam_img_size = 1024

    def __init__(
        self,
        script_args,
        base_image_dir = "data/video_datas/mevis/train/JPEGImages",
        mask_file="data/video_datas/mevis/train/mask_dict.json",
        # expression_file="datasets/video_datas/mevis/train/meta_expressions.json",
        expression_file="/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/DataConstruction/RLdatapipline/generated_folder/MeViS_data.json",
    ):
        super().__init__()
        self.script_args = script_args
        self.expression_file = expression_file
        self.mask_file = mask_file
        self.system_prompt_template = system_prompt_registry[script_args.system_prompt_template]
        self.question_template = question_template_registry[script_args.question_template]
        self.answer_template = answer_template_registry[script_args.answer_template]
        self.image_folder = base_image_dir

        self.dataset_type = script_args.dataset_type
        vid2metaid, mask_dict = self.json_file_preprocess(expression_file, mask_file)
        self.video_infos = vid2metaid
        self.mask_dict = mask_dict

        self.sampled_frames = 5
        self.temporal_sampled_frames = 10
        self.select_number = 1
        self.min_pixel_temporal = 4*28*28
        self.max_pixel_temporal = 32*28*28
        self.min_pixel_spatial = 4*28*28
        self.max_pixel_spatial = 128*28*28


    def __len__(self):
        return len(self.video_infos)

    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)

        if mask_file.endswith('.json'):
            with open(mask_file, 'r') as f:
                mask_dict = json.load(f)
        else:
            raise NotImplementedError

        return expression_datas, mask_dict

    def decode_mask(self, video_masks, image_size, total_frames):
        """
        Modified to handle potentially full video length
        video_masks: list of object masks, where each object mask is a list of frame masks
        total_frames: expected number of frames (either sampled or full)
        """
        ret_masks = []
        for object_masks in video_masks:
            # None object
            if len(object_masks) == 0:
                if len(ret_masks) != 0:
                    _object_masks = ret_masks[0] * 0
                else:
                    _object_masks = np.zeros(
                        (total_frames, image_size[0], image_size[1]), dtype=np.uint8)
            else:
                _object_masks = []
                # Ensure we iterate correctly based on structure
                num_frames_in_mask = len(object_masks[0])

                for i_frame in range(num_frames_in_mask):
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
        ret_masks = ret_masks.flatten(0, 1) # Merge obj and frame dims? Check if this is desired for full video.

        return ret_masks

    def dataset_map_fn(self, data_dict, select_k=5, temporal_k=20):
        images = []
        temporal_images = []

        len_frames = len(data_dict[0]['frames'])
        for object_info in data_dict:
            assert len_frames == len(object_info['frames'])

        # prepare images, random select k frames
        if len_frames > select_k+1:
            # selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
            selected_frame_indexes = np.linspace(0, len_frames-1, num=select_k, dtype=int)
        else:
            # selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
            selected_frame_indexes = np.arange(len_frames)

        if len_frames > temporal_k+1:
            selected_temporal_frame_indexes = np.linspace(0, len_frames-1, num=temporal_k, dtype=int)
        else:
            selected_temporal_frame_indexes = np.arange(len_frames)

        selected_frame_indexes.sort()
        selected_temporal_frame_indexes.sort()

        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_frame_index]
            images.append(os.path.join(self.image_folder, data_dict[0]['video_name'], frame_id + '.jpg'))

        for selected_temporal_frame_index in selected_temporal_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_temporal_frame_index]
            temporal_images.append(os.path.join(self.image_folder, data_dict[0]['video_name'], frame_id + '.jpg'))

        # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]

        # question_text = self.question_template.format(question=exp, total_frames=len_frames)
        question_text = self.question_template.format(question=expressions[0], total_frames=len_frames)

        video_text = f"\nHere are {len(selected_temporal_frame_indexes)} low-resolution frames in this video (frame indice is {str(selected_temporal_frame_indexes.tolist())})\n"
        image_text = f"Here are {len(selected_frame_indexes)} high-resolution frames in this video (frame indice:{str(list(selected_frame_indexes.tolist()))}).\n"

        if len(selected_temporal_frame_indexes)==0: video_text = ""
        if len(selected_frame_indexes)==0: image_text = ""

        content = [{"type": "image"}] * len(selected_temporal_frame_indexes) \
                    + [{"type": "text", "text": video_text}] \
                    + [{"type": "image"}] * len(selected_frame_indexes) \
                    + [{"type": "text", "text": image_text}] \
                    + [{"type": "text", "text": question_text}]

        conversation = json.dumps(
            [
                {"role": "system", "content": self.system_prompt_template},
                {"role": "user", "content": content}
            ]
        )


        video_masks_sampled = []
        video_masks_full = []

        for object_info in data_dict:
            anno_ids = object_info['anno_id']
            obj_masks_sampled = []
            obj_masks_full = []

            for anno_id in anno_ids:
                anno_id = str(anno_id)
                frames_masks = self.mask_dict[anno_id] # This is the full list of masks for the video

                # 4a. Sampled Masks
                frames_masks_sampled_ = []
                for frame_idx in selected_frame_indexes:
                    frames_masks_sampled_.append(copy.deepcopy(frames_masks[frame_idx]))
                obj_masks_sampled.append(frames_masks_sampled_)

                # 4b. Full Masks
                # Deepcopy might be slow for full video, use with caution if video is very long
                # If memory is an issue, we might process on the fly, but for dataset class loading usually okay.
                obj_masks_full.append(frames_masks)

            video_masks_sampled.append(obj_masks_sampled)
            video_masks_full.append(obj_masks_full)

        # read image size from the first image
        first_image_path = images[0]
        first_image = Image.open(first_image_path).convert('RGB')
        if first_image is None:
            return None

        # switch height and width (PIL system (WH vs HW system)
        _image_size = first_image.size
        image_size = (_image_size[1], _image_size[0])

        # Decode Sampled Masks
        masks = self.decode_mask(video_masks_sampled, image_size=image_size, total_frames=len(selected_frame_indexes))

        # Decode Full Masks [New]
        # This will return a tensor covering the entire video duration
        # masks_full = self.decode_mask(video_masks_full, image_size=image_size, total_frames=len_frames)

        ret = {
            'images': images,
            'temporal_images': temporal_images,
            'conversations': conversation,
            'masks': masks,
            'masks_full': video_masks_full, # Pass the full masks out
            'exp': expressions[0],
            'frame_index': selected_frame_indexes.tolist(),
            'temporal_frame_index': selected_temporal_frame_indexes.tolist()
        }
        return ret

    def __getitem__(self, idx):

        index = idx % len(self)
        video_objects_infos = self.video_infos[index]

        total_frames_path = [os.path.join(self.image_folder, video_objects_infos[0]['video_name'], frame_id + '.jpg') for frame_id in video_objects_infos[0]['frames']]

        selected_indexes = [0]
        video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        data_dict = self.dataset_map_fn(video_objects_infos, select_k=self.sampled_frames, temporal_k=self.temporal_sampled_frames)

        mask_existence = video_objects_infos[0]["mask_existence"]

        images_path = data_dict["images"]
        images_ = [Image.open(image).convert('RGB') for image in images_path]
        temporal_images_path = data_dict["temporal_images"]
        temporal_images_ = [Image.open(image).convert('RGB') for image in temporal_images_path]
        width, height = images_[0].size

        spatial_resized_height, spatial_resized_width = smart_resize(height, width, factor=28, min_pixels=self.min_pixel_spatial, max_pixels=self.max_pixel_spatial)
        temporal_resized_height, temporal_resized_width = smart_resize(height, width, factor=28, min_pixels=self.min_pixel_temporal, max_pixels=self.max_pixel_temporal)

        spatial_images = [image.resize((spatial_resized_width, spatial_resized_height)) for image in images_]
        temporal_images = [image.resize((temporal_resized_width, temporal_resized_height)) for image in temporal_images_]

        sam_images_resized = [resize_api(image, (self.sam_img_size, self.sam_img_size)) for image in images_]
        sam_images_np = np.stack([np.array(img) for img in sam_images_resized])
        sam_images = torch.from_numpy(sam_images_np).permute(0, 3, 1, 2)
        sam_images = (sam_images - self.pixel_mean) / self.pixel_std

        prompt = data_dict["conversations"]

        # 1. Process Sampled Masks (For training Loss)
        masks = data_dict["masks"]
        masks = masks.float()
        # Convert to standard binary mask (0 or 1)
        masks = (masks > 0).numpy().astype(np.uint8)

        # 2. Process Full Masks (For Rollout Evaluation)
        masks_full = data_dict["masks_full"]
        # masks_full = (masks_full > 0).numpy().astype(np.uint8)

        return {
            "image": spatial_images,
            "image_index": data_dict["frame_index"],
            "temporal_image": temporal_images,
            "temporal_image_index": data_dict["temporal_frame_index"],
            "problem": data_dict['exp'],
            "solution": [0,0,0,0],
            "prompt": prompt,
            "sam_image": sam_images,
            "image_size": (height, width),
            "mask": masks,       # Sampled masks (e.g. 5 frames)
            "mask_full": masks_full, # ALL masks (e.g. 100 frames) - NEW
            "image_path": images_path,
            "total_frame_path": total_frames_path,
            "total_frames": len(total_frames_path),
            "mask_existence": torch.from_numpy(np.stack(mask_existence)),
        }


class RobustMixedDataset(Dataset):
    def __init__(self, datasets):
        """
        Args:
            datasets (list): datasets to sample from as a single concatenated dataset.
        """
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        self.total_len = sum(self.lens)

        self.cumulative_sizes = np.cumsum(self.lens)

    def __len__(self):
        return int(self.total_len)

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.total_len

        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')

        if dataset_idx == 0:
            inner_idx = idx
        else:
            inner_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][inner_idx]
