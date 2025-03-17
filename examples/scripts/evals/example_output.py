from datasets import load_dataset
from transformers import AutoTokenizer
import os

# 加载数据集

dataset_path = "/nfs100/zhuhao/code_completion/data/aixcoder/base_aixcoder_spm_1123.jsonl" 
# dataset_path = "/nfs100/zhuhao/code_completion/data/aixcoder/base_aixcoder_psm_1123.jsonl"
output_dir = "/nfs100/zhuhao/trl_debug/trl-main/trl-main/examples/scripts/evals"  # 保存路径
output_file = os.path.join(output_dir, "sample_and_tokens_spm.txt")


dataset = load_dataset("json", data_files=dataset_path, split="train")

# 加载 tokenizer
model_name_or_path = "/nfs100/zhuhao/models/aixcoder/aiXcoder-7b-base-weights-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

EOT_TOKEN = "</s>"
def formatting_prompts_func(example):
    prompt = example["prompt"]
    completion = example["output"]
    text = prompt + completion + EOT_TOKEN
    return text

sample = dataset[0]  # 获取第一个样本
formatted_sample = formatting_prompts_func(sample)  # 格式化样本
print("Formatted Sample:")
print(formatted_sample)

# Tokenize 样本
tokenized = tokenizer(formatted_sample, return_tensors="pt", padding=True, truncation=True)
decoded_text = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=False)

print("\nToken IDs:")
print(tokenized["input_ids"])
print("\nDecoded Text:")
print(decoded_text)
# 保存内容到文件

with open(output_file, "w") as f:
    f.write("Formatted Sample:\n")
    f.write(formatted_sample + "\n\n")
    f.write("Token IDs:\n")
    f.write(str(tokenized["input_ids"].tolist()) + "\n\n")
    f.write("Decoded Text:\n")
    f.write(decoded_text + "\n")

print(f"Sample and tokens saved to {output_file}")

# docker run --rm -it --gpus all --net=host \
#     -v /nfs100/zhuhao:/nfs100/zhuhao \
#     -v /nfs100/public:/nfs100/public \
#     trl_env:0910 \
#     python /nfs100/zhuhao/trl_debug/trl-main/trl-main/examples/scripts/evals/example_output.py