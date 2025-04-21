from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import requests
import os
import argparse
import time

parser = argparse.ArgumentParser(description="Run model with specified parameters.")
parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
parser.add_argument("--model_name", type=str, required=True, help="Model name")
parser.add_argument("--model_path", type=str, required=True, help="Model path")
parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt file")
parser.add_argument("--vllm_url", type=str, required=True, help="VLLM service URL")
parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling (default: 0)")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate (default: 1)")

args = parser.parse_args()

print(f"Work Directory: {args.work_dir}")
print(f"Model Name: {args.model_name}")
print(f"Model Path: {args.model_path}")
print(f"Prompt File: {args.prompt_file}")
print(f"VLLM URL: {args.vllm_url}")
print(f"Temperature: {args.temperature}")
print(f"Number of samples: {args.num_samples}")

work_dir = args.work_dir
model_name = args.model_name
model_path = args.model_path
prompt_file = args.prompt_file
vllm_url = args.vllm_url
temperature = args.temperature
n_samples = args.num_samples

save_dir = work_dir
os.makedirs(save_dir, exist_ok=True)

result_file = os.path.join(save_dir, "raw_results.jsonl")
print(f"Result file: {result_file}")

def generate(prompt, temperature, n_samples):
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": temperature,
        "top_p": 0.95,
        "n": n_samples,
    }
    try:
        response = requests.post(vllm_url, json=data)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return [choice.get('text', '') for choice in result['choices']]
            else:
                return ['']
        else:
            print(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
            return ['']
    except Exception as e:
        print(f"请求时发生异常: {e}")
        return ['']

def process_example(example, temperature, n_samples):
    prompt = example["prompt"]
    completions = generate(prompt, temperature, n_samples)
    example["completions"] = completions
    return example

except_list = []
if os.path.exists(result_file):
    with open(result_file, "r") as f:
        for line in f:
            existing_example = json.loads(line)
            except_list.append(existing_example["namespace"])


with open(prompt_file, "r") as f:
    examples = [json.loads(line) for line in f.readlines()]


examples_to_process = [e for e in examples if e["namespace"] not in except_list]
examples_to_process = examples_to_process

st_time = time.time()

with open(result_file, "a+") as ff:

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_example, example, temperature, n_samples): i 
            for i, example in enumerate(examples_to_process)
        }


        results = [None] * len(examples_to_process)

        for future in tqdm(as_completed(futures), total=len(futures)):
            index = futures[future]  
            try:
                result = future.result()
                results[index] = result 
            except Exception as e:
                print(f"处理 namespace '{examples_to_process[index]['namespace']}' 时发生错误: {e}")


        for result in results:
            if result is not None:
                ff.write(json.dumps(result) + "\n")
                ff.flush()  



def expand_completions(result_file_path):
    expanded_results = []
    

    with open(result_file_path, "r") as file:
        for line in file:
            example = json.loads(line)
            namespace = example["namespace"]
            completions = example.get("completions", [])
            

            for i, completion in enumerate(completions):
                new_example = example.copy()  
                new_example["namespace_idx"] = f"{namespace}_{i}"
                new_example["completion"] = completion 
                del new_example["completions"]
                expanded_results.append(new_example)
    
    return expanded_results


expanded_result_file = os.path.join(save_dir, "results.jsonl")

expanded_examples = expand_completions(result_file)


with open(expanded_result_file, "w") as ef:
    for example in expanded_examples:
        ef.write(json.dumps(example) + "\n")

time_file = os.path.join(save_dir, "time.txt")
ed_time = time.time()
with open(time_file, "w") as f:
    f.write(str(ed_time - st_time))
