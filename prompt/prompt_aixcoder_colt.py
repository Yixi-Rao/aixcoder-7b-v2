import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
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

bos_token = "<s>"
eos_token = "</s>"



def get_first_non_empty_indentation(middle):
    """获取middle中第一行有内容的行的缩进"""
    for line in middle.split("\n"):
        if line:
            indent = len(line) - len(textwrap.dedent(line))
            return " " * indent
    return "" 

def get_last_line_indentation(prefix):
    """获取prefix中倒数第一行的缩进"""
    lines = prefix.split("\n")
    for line in reversed(lines):
        if line.strip():
            indent = len(line) - len(textwrap.dedent(line))
            return " " * indent
    return "" 

# Model
max_length = 16384
model_name = "models/aixcoder/aiXcoder-7b-base-weights-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
task_description = (
        "<s><!--the file path is: README.md-->\n"
        "<!--the code file is written by Markdown-->\n"
        "# Task Description:\n"
        "Your task is to complete an incomplete Code file. The input includes in-file context and cross-file context, as described below:\n"
        "# Explanation of Inputs:\n"
        "1. In-file Context:\n"
        "  - Prefix (Context Before): The code that appears before the missing section.\n"
        "  - Suffix (Context Below): The code that follows the missing section.\n"
        "  Your task is to predict the code that should be placed between the prefix and suffix.\n"
        "2. Cross-file Context:\n"
        "  - Global Declarations: Class and API signatures relevant to the current task.\n"
        "  - Similar Code Snippets: Code snippets from the current project that resemble the desired implementation.\n\n"
        "# Input Structure:\n"
        "  - Cross-file Context:\n"
        "    - Global Declarations\n"
        "    - Similar Code Snippets\n"
        "  - In-file Context:\n"
        "    - Suffix\n"
        "    - Prefix\n\n</s>"
    )



def process_sample_instruction_model(sample, language):
    namespace = sample.get('namespace', '')
    prefix = sample.get('prefix', '')
    suffix = sample.get('suffix', '')
    middle = sample.get('middle', '')
    function_name = sample.get('function_name', '')
    function_prefix = sample.get('function_prefix', '')
    cross_file_dependency = sample.get('cross_file_dependency', [])
    similar_functions = sample.get('similar_functions', [])
    code_file_path = sample.get('code_file_path', '')

    comment_symbol = LANGUAGE_COMMENT_MAP.get(language)

    global_declarations = f"{comment_symbol} Global Declarations:\n"
    similar_code_snippets = f"{comment_symbol} Similar Code Snippets:\n"

    cross_file_context = global_declarations + similar_code_snippets

    fim_format = _format(prefix, code_file_path, language)

    fim_task = f"{comment_symbol} Fill-in-the-Middle Task\n\n"
    fim_task += bos_token + "▁<AIX-SPAN-PRE>▁<AIX-SPAN-POST>"
    fim_task += f"{suffix}▁<AIX-SPAN-MIDDLE>"
    fim_task += fim_format

    middle_indentation = get_last_line_indentation(prefix)
    comment_block = f"{middle_indentation}{comment_symbol} The completed code:\n"

    fim_task += comment_block

    prompt = bos_token + task_description + eos_token + cross_file_context + fim_task

    middle_token_length = len(tokenizer(middle, truncation=False, add_special_tokens=False)['input_ids'])
    prompt_token_length = len(tokenizer(prompt, truncation=False, add_special_tokens=False)['input_ids'])
    output_tokens_length = middle_token_length * 2
    remaining_length = max_length - output_tokens_length - prompt_token_length

    i = 0
    cb, token_length_sum = 0, 0
    max_itr_len = max(len(cross_file_dependency), len(similar_functions))

    if max_itr_len == 0:
        cross_file_context = f"{comment_symbol} Global Declarations:\nno related global declarations\n"
        cross_file_context += f"{comment_symbol} Similar Code Snippets:\nno related similar code snippets\n"
        prompt = bos_token + task_description + eos_token + cross_file_context + fim_task
        return prompt, middle, remaining_length, i, token_length_sum, cb

    for i in range(1, max_itr_len + 1):
        global_decl = cross_file_dependency[:i] if i < len(cross_file_dependency) else cross_file_dependency
        similar_func = similar_functions[:i] if i < len(similar_functions) else similar_functions
        similar_func.reverse()

        global_declarations = ''
        for dep in global_decl:
            code_path = dep.get('code_file_path', '')
            abstraction = dep.get('abstraction', '')
            if abstraction.strip():
                scf = bos_token + _format(abstraction, code_path, language) + eos_token
                global_declarations += scf

        similar_code_snippets = ''
        for snippet in similar_func:
            code_path = snippet.get('code_file_path', '')
            code = snippet.get('code_block', '')
            scf = bos_token + _format(code, code_path, language) + eos_token
            similar_code_snippets += scf

        if global_declarations == '':
            global_declarations = 'no related global declarations\n'
        if similar_functions == []:
            similar_code_snippets = 'no related similar code snippets\n'

        global_declarations = f"{comment_symbol} Global Declarations:\n" + global_declarations
        similar_code_snippets = f"{comment_symbol} Similar Code Snippets:\n" + similar_code_snippets
        token_length_sum = len(tokenizer(global_declarations, truncation=False, add_special_tokens=False)['input_ids']) + len(tokenizer(similar_code_snippets, truncation=False, add_special_tokens=False)['input_ids'])

        if remaining_length - token_length_sum < 0:
            break

        cross_file_context = global_declarations + similar_code_snippets
        prompt = task_description + cross_file_context + fim_task

    cb = comment_block + middle

    return prompt, middle, remaining_length, i, token_length_sum, cb

output_dir = f"/data/aixcoder_our_style"
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
    input_file = f"test_data/{file_name}"
    output_file = f"{output_dir}/{file_name}"

    with open(input_file, 'r', encoding='utf-8') as f:
        total_samples = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as f:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for line_num, line in enumerate(tqdm(f, desc=f"Processing {file_name} samples", total=total_samples), 1):
                try:
                    sample = json.loads(line)
                    prompt, middle, remaining_length, itr, token_length_sum, fim_task = process_sample_instruction_model(sample, language)
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





