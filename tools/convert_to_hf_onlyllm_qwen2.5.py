import argparse
import os.path as osp
import torch
from mmengine.dist import master_only
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Export inner Qwen2.5-VL model for vLLM (BF16)')
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

    # 1. 解析配置
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # 2. 构建完整 Sa2VA 模型
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    
    # 3. 加载权重
    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)
    else:
        state_dict = torch.load(args.pth_model, map_location='cpu', weights_only=False)

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')
    
    # 4. 合并 LoRA
    model._merge_lora()
    
    # ================= 关键步骤：提取内部模型 =================
    # vLLM 无法识别 Sa2VAChatModel 的外壳，必须提取 model.mllm.model
    # 这对应原生的 Qwen2.5-VL 结构
    target_model = model.mllm.model
    print(f"Extracted inner model type: {type(target_model)}")

    # ================= Fix: Sync Vocab Size =================
    # Fix for vLLM error: assert loaded_weight.shape[output_dim] == self.org_vocab_size
    # 检查权重中的实际词表大小，并更新 config
    if hasattr(target_model, 'get_input_embeddings'):
        embedding_weight = target_model.get_input_embeddings().weight
        actual_vocab_size = embedding_weight.shape[0]
        if hasattr(target_model, 'config') and target_model.config.vocab_size != actual_vocab_size:
            print(f"Warning: Config vocab size ({target_model.config.vocab_size}) mismatch with weight size ({actual_vocab_size}).")
            print(f"Updating config.vocab_size to {actual_vocab_size} to prevent vLLM loading errors.")
            target_model.config.vocab_size = actual_vocab_size

    # ================= 关键步骤：转为 BF16 =================
    print("Converting model to bfloat16 for vLLM efficiency...")
    target_model.to(torch.bfloat16)

    # 5. 准备保存路径
    iter_str = os.path.basename(args.pth_model).split('.')[0]
    if args.save_path is None:
        args.save_path = f"./{os.path.dirname(args.pth_model)}_{iter_str}_vllm_bf16"

    # 7. 清洗 Config (让 vLLM 认为这是原生 Qwen2.5-VL)
    if hasattr(target_model, 'config'):
        # 强制指定标准架构，解决 vLLM 不识别的问题
        target_model.config.architectures = ["Qwen2_5_VLForConditionalGeneration"]
        
        # 移除自定义代码映射，防止 vLLM 尝试加载不存在的本地文件
        if hasattr(target_model.config, 'auto_map'):
            del target_model.config.auto_map
            
        # ================= Fix: Remove text_config =================
        # 原生 Qwen2.5-VL 是一体化配置，不需要 text_config 字段
        if hasattr(target_model.config, 'text_config'):
            print("Removing non-standard 'text_config' from configuration...")
            del target_model.config.text_config

        # Final sanity check on config vocab size
        current_weight_shape = target_model.get_input_embeddings().weight.shape[0]
        if target_model.config.vocab_size != current_weight_shape:
             print(f"Final Config Update: Setting config.vocab_size to {current_weight_shape}")
             target_model.config.vocab_size = current_weight_shape

    # 7. 保存
    print(f"Saving stripped model to {args.save_path}...")
    
    # 保存模型权重 (bf16)
    target_model.save_pretrained(args.save_path, safe_serialization=True)
    
    # 保存 Processor 和 Tokenizer (这对 vLLM 处理图像输入至关重要)
    if hasattr(model.mllm, 'processor'):
        model.mllm.processor.save_pretrained(args.save_path)
    
    if hasattr(model.mllm, 'tokenizer'):
        model.mllm.tokenizer.save_pretrained(args.save_path)

    master_print(f"\nSuccess! Model saved to: {args.save_path}")
    master_print("You can now load this path directly with vLLM.")

if __name__ == '__main__':
    main()