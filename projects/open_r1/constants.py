from projects.open_r1.rewards import *
from projects.open_r1.rewards_dense import *

# REWARD MAPING
reward_funcs_registry = {
    "pr1_grounding": pr1_grounding_reward,
    "pr1_grounding_format": pr1_grounding_format_reward,
    "pr1_grounding_format_max_0p1": pr1_grounding_format_reward_max_0p1,
    "pr1_grounding_format_reason": pr1_grounding_format_reward_reason,
    "think_format": think_format_reward,
    "think_select_format": think_select_format_reward,
    "vtg_keyframe": vtg_keyframe_reward_cot,
    "vtg_iou": vtg_iou_reward_cot,
    "pr_maskiou": pr_maskiou_reward,
    "pr_keyframe_valid": pr_keyframe_valid_reward,
    "mr_progressive_maskiou": mr_progressive_maskiou_reward,
    "strict_format_cot": strict_format_reward_cot,
    "step_format_cot": step_format_reward_cot,
    "mr_progressive_keyframe": mr_progressive_keyframe_reward,
    "lr_keyframeiou": lr_keyframe_reward,
    "lr_maskiou": lr_maskiou_reward,
    "lr_vtg_iou": lr_vtg_iou_reward,
    "lr_vtg_keyframe": lr_vtg_keyframe_reward,
    "lr_keyframe_vs_best_mask": lr_keyframe_vs_best_mask_reward,
    "lr_keyframe_vs_mask": lr_keyframe_vs_mask_reward,
    "lr_progressive_keyframe": lr_progressive_keyframe_reward,

    "lr_keyframeiou_dense": lr_keyframe_reward_dense,
    "lr_maskiou_dense": lr_maskiou_reward_dense,
    "lr_vtg_iou_dense": lr_vtg_iou_reward_dense,
    "lr_keyframe_vs_mask_dense": lr_keyframe_vs_mask_reward_dense,
    "lr_progressive_keyframe_dense": lr_progressive_keyframe_reward_dense,


    "simple_lr_maskiou": simple_lr_maskiou_reward,
    "simple_lr_keyframe": simple_lr_keyframe_reward,

    "termination": termination_reward,
}

# SYSTEM PROMPTS
LLAVA_SYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

QWEN2_SYS = (
    "You are a helpful assistant. "
)

R1V_SYS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

SEG_COT_SYS = (
    "You are a helpful assistant. Your ultimate goal is to perform video temporal grounding (VTG) and referring video object segmentation (RefVOS). " 
    "Based on the text query and video content, output your thinking process within <think> and </think> tags. " 
    "If anything is unclear, you can select frames from the video for a clearer view by outputting <select>{\"start\": start, \"end\": end, \"keyframe\": keyframe}</select>, where 'start' and 'end' are the start and end frame indices of the region for detailed analysis, and 'keyframe' is the corresponding keyframe index. " 
    "Once the final answer is confirmed, provide it within <answer><VTG>{\"start\": start, \"end\": end, \"keyframe\": keyframe}</VTG>, <RefSeg></RefSeg></answer>."
)

system_prompt_registry = {
    "default": QWEN2_SYS,
    "llava": LLAVA_SYS,
    "qwen": QWEN2_SYS,
    "r1v": R1V_SYS,
    "seg_cot": SEG_COT_SYS,
}

SAMR1_V2_TEMPLATE = """Analyze the image and locate objects: "{question}".
    Please always follow this format:
    1. First, put your thinking precess in <think></think> tags.
    2. Then provide the grounding result using the specified format.
    For example, "<think>think process here</think>\n<|object_ref_start|>object<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>"."""


SAMR1_V3_TEMPLATE = """Analyze the image and locate objects: "{question}".
    Please always follow this format:
    1. First, put your thinking precess in <think></think> tags.
    2. Then provide the grounding result using the specified format.
    For example:
    <think>I first searched for the object based on its shape and location in the image ...</think>
    <|object_ref_start|>apple<|object_ref_end|><|box_start|>(100,120),(180,200)<|box_end|>
    """

SAMR1_V4_TEMPLATE = \
    "Locate \"{question}\", report the bbox coordinates in JSON format." \
    "Compare the difference between objects and find the most closely matched one." \
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    "Output the one bbox inside the interested object in JSON format." \
    "i.e., <think>thinking process here</think>" \
    "<answer>answer here</answer>"

QWEN2P5_TEMPLATE = '''
Output the bounding box of the {question} in the image.
i.e."<|object_ref_start|>object<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>".
'''

COT_TEMPLATE = '''
Find the object that best matches {question} and provide its bounding box.
Please:
1. Analyze all objects in the image carefully
2. Compare candidates against the target description
3. Select the most closely matching object
4. Provide precise bounding box coordinates
Format your response as:
<think>
[Your step-by-step analysis and reasoning]
</think>
<answer>
```json
[
	{{"bbox_2d": [x1, y1, x2, y2], "label": {question}}}
]
```
</answer>
'''


SEG_TEMPLATE = '''
Please segment {question}
'''

SEG_COT_TEMPLATE = '''
This video totally has {total_frames} frames, The indice of the sampled frames is {sample_indice}. The text query is \"{question}\". Show me your thought process for finding and segmenting this target.
'''

SEG_COT_TEMPORAL_TEMPLATE = '''
The query is \"{question}\". Show me your thought process for finding and segmenting this target. This video has {total_frames} frames.
'''


question_template_registry = {
    "default": "{question}",
    "pr1_grounding": "Output the bounding box of the {question} in the image.",
    "samr1": """Please find the bounding box (bbox) containing "{question}".
            Please compare the differences between objects, locate, and identify the most closely matched target object.
            Please first output the thinking process in the "<think>...</think>" tag pair, then output the bounding box that exactly contains the most closely matched target object at the next line.
            For example, "<think>(here is the thinking process)</think>\n<|object_ref_start|>(the most closely matched target object)<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>".""",
    "samr1_v2": SAMR1_V2_TEMPLATE,
    "samr1_v3": SAMR1_V3_TEMPLATE,
    "qwen2p5": QWEN2P5_TEMPLATE,
    "samr1_v4": SAMR1_V4_TEMPLATE,
    "cot_v1": COT_TEMPLATE,
    "seg_grounding": SEG_TEMPLATE,
    "seg_cot": SEG_COT_TEMPLATE,
    "seg_temporal_cot": SEG_COT_TEMPORAL_TEMPLATE,
}
 
answer_template_registry = {
    "default": "{answer}",
    "r1v": "<answer> {answer} </answer>"
}