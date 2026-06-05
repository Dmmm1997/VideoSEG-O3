from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_dir = "work_dirs/sa2va_qwen25_cot_coldstartv3/hf_model_llm"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_dir)

print(model.config.model_type)
print(model.config.architectures)
print(model.config.hidden_size)  # 看看是不是正常值（比如 4096/5120 这种）
