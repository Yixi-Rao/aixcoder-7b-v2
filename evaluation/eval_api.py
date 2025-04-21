import re
import os
import json
import nltk.translate.bleu_score as bleu
from codebleu import calc_codebleu
from nltk.translate.bleu_score import SmoothingFunction
import editdistance
from utils import tokenize_for_bleu_eval, truncate, dpo_deduplicate_jsonl

import argparse
import logging
from tqdm import tqdm




def deduplicate_samples(samples, max_samples=3):
    seen_completions = set()
    valid_samples = []

    for sample in samples:
        completion = sample.get("completion")
        ground_truth = sample.get("output")
        
        if ground_truth is None or completion is None:
            continue

        if completion == ground_truth:
            continue
        if completion in ground_truth:
            continue

        if completion not in seen_completions:
            seen_completions.add(completion)
            valid_samples.append(sample)

    return valid_samples[:max_samples] if len(valid_samples) >= max_samples else valid_samples


logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description="Run model with specified parameters.")
parser.add_argument("--work_dir", type=str, required=True, help="Working directory")
parser.add_argument("--model_name", type=str, required=True, help="Model name")
parser.add_argument("--model_path", type=str, required=True, help="Model path")
parser.add_argument("--seed_data", type=str, required=True, help="Api import data")
parser.add_argument("--bleu_threshold", type=float, default=0.5, help="BLEU_threshold for filter")

args = parser.parse_args()
print(f"Work Directory: {args.work_dir}")
print(f"Model Name: {args.model_name}")

work_dir = args.work_dir
model_name = args.model_name
model_path = args.model_path
seed_data_file = args.seed_data
bleu_threshold = args.bleu_threshold

in_file = False

merge_file = os.path.join(work_dir, model_name, "api_results.jsonl")
final_file = os.path.join(work_dir, model_name, "api_final_results.jsonl")
save_dir = os.path.join(work_dir, model_name, "api")

if in_file:
    result_file = os.path.join(save_dir, "results_infile.jsonl")
else:
    result_file = os.path.join(save_dir, "results.jsonl")

import pdb
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tot_completion_token_len = 0
exact_match = 0
bleu_score = 0
length = 0
tot = 0
distance = 0
code_bleu = 0

namespace_samples = {}

with open(result_file, "r") as f:
    for line in f.readlines():
        example = json.loads(line)
        namespace = example.get("namespace")  
        namespace += example.get("output", "")

        if namespace not in namespace_samples:
            namespace_samples[namespace] = []

        namespace_samples[namespace].append(example)


count = 0
for namespace, samples in tqdm(namespace_samples.items()):
    unique_samples = deduplicate_samples(samples)
    for example in samples:
        length += 1
        code = example["completion"]
        tot_completion_token_len += len(tokenizer(example["completion"], truncation=False, add_special_tokens=False)['input_ids'])
        ground_truth = example["output"]

        if code == ground_truth:
            exact_match += 1
            example["exact_match"] = 1
        else:
            example["exact_match"] = 0
        
        result = calc_codebleu([ground_truth], [code], lang="python", weights=(1, 0, 0, 0), tokenizer=tokenize_for_bleu_eval)
        example["ngram_match_score"] = result["ngram_match_score"]
        bleu_score += result["ngram_match_score"]

        example["codebleu"] = result["codebleu"]
        code_bleu += result["codebleu"]

        code_tokens = tokenize_for_bleu_eval(code)
        ground_truth_tokens = tokenize_for_bleu_eval(ground_truth)

        example["editdistance"] = editdistance.eval(code_tokens, ground_truth_tokens)
        distance += example["editdistance"]

    if len(unique_samples) > 1:
        for example in unique_samples:
            with open(merge_file, "a") as f_out:
                count += 1
                f_out.write(json.dumps(example) + "\n")

exact_match_score = exact_match / length
bleu_avg_score = bleu_score / length
code_bleu_avg = code_bleu / length
avg_edit_distance = distance / length
avg_token_len = tot_completion_token_len / length

print(f"Exact Match: {exact_match_score:.4f}")
print(f"BLEU Score: {bleu_avg_score:.4f}")
print(f"Code BLEU: {code_bleu_avg:.4f}")
print(f"Average Edit Distance: {avg_edit_distance:.4f}")
print(f"Average Token Length: {avg_token_len:.4f}")
print("All Saved Unique samples:", count)


