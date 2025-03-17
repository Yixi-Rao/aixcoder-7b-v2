# aiXcoder-colt

This repository contains the implementation of CoLT in our papers, a novel fine-tuning approach for enhancing large language models' ability to utilize information within long contexts for code completion tasks.

## Resources

### Dataset
- **CoLT-132K dataset**: A large-scale repo-level code completion dataset comprising 132,000 samples across four programming languages.
- **Download**: [https://zenodo.org/records/15019938](https://zenodo.org/records/15019938)

### Models
- **Trained Models**: This repository includes our models (aiXcoder-7B-v2, DS-Coder and Code Llama) trained with the CoLT approach.
- **Download**: [https://zenodo.org/records/15021246](https://zenodo.org/records/15021246)

## Experiment Reproduction

We use the latest TRL framework(https://github.com/huggingface/trl) code for our experiments. To reproduce our results:

### 1. Prompt Construction Files
- `aiXcoder-colt/prompt/prompt_aixcoder_colt.py`
- `aiXcoder-colt/prompt/prompt_codellama.py`
- `aiXcoder-colt/prompt/prompt_codeqwen.py`
- `aiXcoder-colt/prompt/prompt_deepseekcoder.py`

### 2. Training Scripts
- **SFT (Supervised Fine-Tuning)**: See scripts in `aiXcoder-colt/commands/sft/`
- **DPO (Direct Preference Optimization)**: See scripts in `aiXcoder-colt/commands/po/`

### 3. Reject Sampling for DPO

The `aiXcoder-colt/Reject_Sample/` directory contains implementation and evaluation scripts for our reject sampling approach used in Direct Preference Optimization:

- **Model-specific implementations**:
  - `aixcoder/`:  Reject sampling for aiXcoder model
  - `codellama/`: Reject sampling for Code Llama model
  - `deepseek/`: Reject sampling for DeepSeek-Coder model

- **Evaluation scripts**:
  - `eval_api.py`: API-based evaluation script
  - `eval_line.py`: Line-level evaluation script
  - `eval_span.py`: Span-level evaluation script
  - `inference.py`: Model inference script for generating completions

## Dependencies

In our experiments, we utilized two Docker environments for TRL training and vLLM (reject sampling). Below are the key dependencies for each environment, excluding redundant packages:

### TRL Training Environment:
- transformers==4.46.0.dev0
- torch==2.4.0a0+07cecf4168.nv24.5
- accelerate==1.0.0
- deepspeed==0.15.2
- peft==0.13.1
- flash-attn==2.4.2
- datasets==3.0.1
- wandb==0.15.0

### vLLM Inference Environment:
- vllm==0.6.0+cu124
- torch==2.4.0
- transformers==4.44.2
- vllm-flash-attn==2.6.1
- xformers==0.0.27.post2
- flashinfer==0.1.6+cu121torch2.4
- fastapi==0.114.1
- uvicorn==0.30.6

Complete dependency lists can be found in the `dependency` directory.