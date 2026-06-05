from projects.sa2va.hf.models_qwen3vl.modeling_sa2va_qwen_cotv2 import Sa2VAChatModelQwenCOTV2
import torch

# model_path_1 = "/root/paddlejob/workspace/env_run/daiming/project/LENS/outputs/qwen3_rl_datarecons_seglogps_new/checkpoint-191"
# model_path_1 = "/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/VideoRLSEG/outputs/qwen3_rl_datarecons/checkpoint-191"
model_path_1 = "/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/VideoRLSEG/outputs/qwen3_rl_withsegloss_datarecons/checkpoint-191"

model_1 = Sa2VAChatModelQwenCOTV2.from_pretrained(
        model_path_1,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()

sam2_1 = model_1.grounding_encoder.sam2_model


model_path_2 = "/root/paddlejob/workspace/env_run/daiming/project/MyPub/VideoRL/Sa2VA/work_dirs/sa2va_qwen_cot_coldstartv3/hf_model"

model_2 = Sa2VAChatModelQwenCOTV2.from_pretrained(
        model_path_2,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()

sam2_2 = model_2.grounding_encoder.sam2_model



# Compare the state dictionaries of the sam2_model from both models
sam2_1_weights = sam2_1.state_dict()
sam2_2_weights = sam2_2.state_dict()

# Check if the keys (parameters) are the same in both models
same_keys = sam2_1_weights.keys() == sam2_2_weights.keys()

# Compare the actual weights
weights_are_same = all(torch.allclose(sam2_1_weights[key], sam2_2_weights[key]) for key in sam2_1_weights)

print(f"Are the keys of the sam2 models the same? {same_keys}")
print(f"Are the weights of the sam2 models identical? {weights_are_same}")


# List to store keys where weights are different
different_keys = []

# Compare the weights for each key
for key in sam2_1_weights:
    if not torch.allclose(sam2_1_weights[key], sam2_2_weights[key]):
        different_keys.append(key)

# Output the keys that have different weights
if different_keys:
    print("The following keys have different weights:")
    for key in different_keys:
        print(key)
else:
    print("All weights are the same.")