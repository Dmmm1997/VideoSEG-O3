import argparse
import os
import os.path as osp
import torch
import re
import json
from mmengine.dist import master_only
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend

def parse_args():
    parser = argparse.ArgumentParser(description='Convert XTuner Qwen3VL to vLLM-Compatible Format')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth-model', help='pth model file')
    parser.add_argument(
        '--save-path', type=str, default=None, help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    # 1. 加载配置
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')
    
    cfg = Config.fromfile(args.config)
    
    print(">>> Building Model from Config...")
    model = BUILDER.build(cfg.model)

    # 2. 加载权重
    print(f">>> Loading weights from {args.pth_model}...")
    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)
    else:
        state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # 宽松加载，因为我们要剥离一部分
    model.load_state_dict(state_dict, strict=False)

    # 合并 LoRA
    print(">>> Merging LoRA weights...")
    model._merge_lora()

    # 3. 提取核心原生模型
    # 根据你提供的代码：Qwen3VL wrapper 内部是 self.model = Qwen3VLForConditionalGeneration
    # 而 XTuner 通常把 wrapper 放在 model.mllm 中
    try:
        native_model = model.mllm.model
    except AttributeError:
        # 备用方案：如果层级不同
        native_model = model.model_

    native_model.to(torch.bfloat16)
    
    target_class_name = native_model.__class__.__name__
    print(f">>> Extracted Native Model: {target_class_name}")

    # 4. 获取 Tokenizer / Processor
    if hasattr(model.mllm, 'processor'):
        processor = model.mllm.processor
        tokenizer = processor.tokenizer
    else:
        tokenizer = model.mllm.tokenizer
        processor = None

    # ==========================================
    # 核心修复：vLLM 兼容性 (Config Flattening)
    # ==========================================
    print("\n>>> Applying vLLM Compatibility Patches...")
    
    # 1. 清理 AutoMap (防止加载自定义 Sa2VA 代码)
    if hasattr(native_model.config, "auto_map"):
        print("   - Removing auto_map")
        del native_model.config.auto_map
    
    # 2. 强制指定架构名
    if hasattr(native_model.config, "architectures"):
        native_model.config.architectures = [target_class_name]

    # 3. 将 text_config 参数扁平化到根目录
    # 这是解决 'AttributeError: ... has no attribute hidden_size' 的关键
    if hasattr(native_model.config, 'text_config'):
        # 确保 vocab_size 最新
        native_model.config.text_config.vocab_size = len(tokenizer)
        
        # 需要复制给 vLLM 看的字段
        vllm_required_keys = [
            'vocab_size',
            'hidden_size',
            'num_hidden_layers',
            'num_attention_heads',
            'num_key_value_heads',
            'intermediate_size',
            'rms_norm_eps',
            'max_position_embeddings',
            'rope_theta',
            'sliding_window' # 如果有的话
        ]
        
        text_cfg = native_model.config.text_config
        mapped_count = 0
        for key in vllm_required_keys:
            if hasattr(text_cfg, key):
                val = getattr(text_cfg, key)
                setattr(native_model.config, key, val)
                mapped_count += 1
        
        print(f"   - Flattened {mapped_count} keys from text_config to root config (hidden_size, etc.)")
    else:
        # 如果本身就是扁平的，直接更新 vocab_size
        native_model.config.vocab_size = len(tokenizer)

    # 4. 处理 Jinja Template
    if 'template' in cfg:
        template_str = cfg.template
        # 移除 System Prompt 注入逻辑，保持纯净
        system_prompt_pattern = re.compile(
            r"{% if loop\.first and message\['role'] != 'system' %}.*?<\|im_end\|>\s*{% endif %}",
            re.DOTALL
        )
        template_str = system_prompt_pattern.sub('', template_str)
        tokenizer.chat_template = template_str
        print("   - Cleaned chat_template")

    # ==========================================
    # 保存模型
    # ==========================================
    iter_str = os.path.basename(args.pth_model).split('.')[0]
    if args.save_path is None:
        args.save_path = f"./{os.path.dirname(args.pth_model)}_{iter_str}_vllm_ready"

    print(f"\n>>> Saving to {args.save_path} ...")
    
    # 保存模型 (此时 config 已经被修改过)
    native_model.save_pretrained(args.save_path)
    
    # 保存 Processor / Tokenizer
    if processor is not None:
        processor.save_pretrained(args.save_path)
    else:
        tokenizer.save_pretrained(args.save_path)

    # 最后的验证：手动检查保存后的 json
    config_path = os.path.join(args.save_path, "config.json")
    with open(config_path, 'r') as f:
        saved_cfg = json.load(f)
    
    print("\n>>> Verification:")
    if "hidden_size" in saved_cfg:
        print(f"   [OK] 'hidden_size' found in root config: {saved_cfg['hidden_size']}")
    else:
        print("   [WARNING] 'hidden_size' NOT found in root config!")

    print(f"   [OK] Architectures: {saved_cfg.get('architectures')}")

    master_print("\n>>> Done! You can now load this with vLLM.")

if __name__ == '__main__':
    main()