import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset,  load_from_disk
from tqdm import tqdm
import re
from utils import _format
import textwrap

LANGUAGE_COMMENT_MAP = {
    "python": "#",
    "c++": "//",
    "java": "//",
    "go": "//"
}

def get_first_non_empty_indentation(middle):
    """获取middle中第一行有内容的行的缩进"""
    for line in middle.split("\n"):
        if line:
            indent = len(line) - len(textwrap.dedent(line))
            return " " * indent
    return "" 

def get_last_line_indentation(prefix):
    """获取prefix中倒数第一行的缩进"""
    # 按行拆分prefix
    lines = prefix.split("\n")
    
    # 从最后一行开始向前找，直到找到有内容的行
    for line in reversed(lines):
        if line.strip():  # 检查非空行
            indent = len(line) - len(textwrap.dedent(line))
            return " " * indent
    return ""  # 如果没有找到非空行，返回空字符串



# Model
model_name = "models/CodeLlama-7b-base-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# <PRE> {prefix} <SUF>{suffix} <MID>
prefix_token = "▁<PRE>"
suffix_token = "▁<SUF>"
mid_token = "▁<MID>"
eot_token = "▁<EOT>"

def raw_format(s, path, language):
    comment_symbol = LANGUAGE_COMMENT_MAP.get(language)
    prompt = comment_symbol + " the below code fragment can be found in:\n" + comment_symbol + " " + path + "\n" + "-" * 50 + "\n"
    # 给s每行前加上"# "，并在每行后加上"\n"
    lines = s.split("\n")
    for line in lines:
        prompt += comment_symbol + " " + line + "\n"
    prompt += "-" * 50 + "\n"
    return prompt

def process_sample_instruction_model(sample, language, max_length):
    namespace = sample.get('namespace', '')
    prefix = sample.get('prefix', '')
    suffix = sample.get('suffix', '')
    middle = sample.get('middle', '')
    function_name = sample.get('function_name', '')
    function_prefix = sample.get('function_prefix', '')
    cross_file_dependency = sample.get('cross_file_dependency', [])
    similar_functions = sample.get('similar_functions', [])
    project_dir = sample.get('project_dir','')
    code_file_path = sample.get('code_file_path','')
    
    comment_symbol = LANGUAGE_COMMENT_MAP.get(language)

    # Construct the Fill-in-the-Middle Task
    fim_task = "# Fill-in-the-Middle Task\n\n"
    fim_task += f"{prefix_token}{prefix} "
    fim_task += f"{suffix_token}{suffix} {mid_token}"

    prompt = fim_task

    meta_prompt = comment_symbol + " Here are some relevant code fragments fromRetrievedother files of the repo:\n" + '-' * 50 + "\n"
    comment_block = comment_symbol + " Here are some relevant code fragments fromRetrievedother files of the repo:\n" + "-" * 50 + "\n"
    middle_token_length = len(tokenizer(middle, truncation=False, add_special_tokens=False)['input_ids'])
    prompt_token_length = len(tokenizer(prompt, truncation=False, add_special_tokens=False)['input_ids'])
    output_tokens_length = middle_token_length * 2
    remaining_length = max_length - output_tokens_length - prompt_token_length

    i = 0
    cb, token_length_sum = 0, 0
    max_itr_len = max(len(cross_file_dependency), len(similar_functions))


    for i in range(1, max_itr_len + 1):
        global_decl = cross_file_dependency[:i] if i < len(cross_file_dependency) else cross_file_dependency
        similar_func = similar_functions[:i] if i < len(similar_functions) else similar_functions
        similar_func.reverse()

        global_declarations = ''
        for dep in global_decl:
            code_path = dep.get('code_path', '')
            code = dep.get('abstraction', '')
            if code.strip():  # Check if the abstraction is not empty
                global_declarations += raw_format(code, code_path, language)

        similar_code_snippets = ''
        for snippet in similar_func:
            code_path = snippet.get('code_file_path', '')
            code = snippet.get('code_block', '')
            if code.strip():  # Check if the code is not empty
                similar_code_snippets += raw_format(code, code_path, language)
        
        token_length_sum = len(tokenizer(global_declarations, truncation=False, add_special_tokens=False)['input_ids']) +  len(tokenizer(similar_code_snippets, truncation=False, add_special_tokens=False)['input_ids'])
        if remaining_length - token_length_sum < 0:
            break
        cross_file_context =  global_declarations + similar_code_snippets
        comment_block = meta_prompt + cross_file_context


    ret = comment_block + fim_task

    return ret, middle



max_length = 16384
output_dir = f"data/codellama"
os.makedirs(output_dir, exist_ok=True)

files_to_process = [
    "cplus_api.jsonl", "cplus_line.jsonl", "cplus_structured_span.jsonl",
    "go_api.jsonl", "go_line.jsonl", "go_structured_span.jsonl",
    "java_api.jsonl", "java_line.jsonl", "java_structured_span.jsonl",
    "python_api.jsonl", "python_line.jsonl", "python_structured_span.jsonl"
]

for file_name in files_to_process:
    language = file_name.split("_")[0]
    if language == "cplus":
        language = "c++"
    comment_symbol = LANGUAGE_COMMENT_MAP.get(language)
    input_file = f"CoLT-132K/sft_data/{file_name}"
    output_file = f"{output_dir}/{file_name}"

    with open(input_file, 'r', encoding='utf-8') as f:
        total_samples = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as f:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for line_num, line in enumerate(tqdm(f, desc=f"Processing {file_name} samples", total=total_samples), 1):
                try:
                    sample = json.loads(line)
                    prompt, middle = process_sample_instruction_model(sample, language, max_length)
                    if prompt == -1:
                        continue
                    output = middle
                    data_entry = {
                        'prompt': prompt,
                        'output': output,
                        'namespace': sample['namespace'],
                    }
                    out_f.write(json.dumps(data_entry) + '\n')
                except json.JSONDecodeError as e:
                    print(f'Error decoding JSON on line {line_num}: {e}')
                except Exception as e:
                    print(f'Error processing sample on line {line_num}: {e}')




output_dir = f"data/codellama_po"
os.makedirs(output_dir, exist_ok=True)

files_to_process = [
    "cplus_api.jsonl", "cplus_line.jsonl", "cplus_structured_span.jsonl",
    "go_api.jsonl", "go_line.jsonl", "go_structured_span.jsonl",
    "java_api.jsonl", "java_line.jsonl", "java_structured_span.jsonl",
    "python_api.jsonl", "python_line.jsonl", "python_structured_span.jsonl"
]

for file_name in files_to_process:
    language = file_name.split("_")[0]
    if language == "cplus":
        language = "c++"
    comment_symbol = LANGUAGE_COMMENT_MAP.get(language)
    input_file = f"CoLT-132K/po_data/{file_name}"
    output_file = f"{output_dir}/{file_name}"

    with open(input_file, 'r', encoding='utf-8') as f:
        total_samples = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as f:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for line_num, line in enumerate(tqdm(f, desc=f"Processing {file_name} samples", total=total_samples), 1):
                try:
                    sample = json.loads(line)
                    prompt, middle = process_sample_instruction_model(sample, language, max_length)
                    if prompt == -1:
                        continue
                    output = middle
                    data_entry = {
                        'prompt': prompt,
                        'output': output,
                        'namespace': sample['namespace'],
                    }
                    out_f.write(json.dumps(data_entry) + '\n')
                except json.JSONDecodeError as e:
                    print(f'Error decoding JSON on line {line_num}: {e}')
                except Exception as e:
                    print(f'Error processing sample on line {line_num}: {e}')
