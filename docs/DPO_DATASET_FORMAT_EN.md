# DPO Dataset Format Specification

DPO (Direct Preference Optimization) is a training method that directly optimizes human preferences. This document details the format requirements for DPO datasets used in this project.

## Core Fields

DPO datasets must contain the following three core fields:

- **`prompt`**: Input prompt, can be a string or conversational format
- **`chosen`**: Preferred response/completion content  
- **`rejected`**: Rejected response/completion content

## Data Format Types

### 1. Simple String Format

The most basic format where all fields are simple strings:

```json
{
  "prompt": "Implement a function to calculate Fibonacci numbers",
  "chosen": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "rejected": "def fib(n):\n    return n + n - 1"
}
```

### 2. Conversational Format

Suitable for chat scenarios, where each field is a list of messages:

```json
{
  "prompt": [
    {"role": "user", "content": "What color is the sky?"}
  ],
  "chosen": [
    {"role": "assistant", "content": "The sky is blue."}
  ],
  "rejected": [
    {"role": "assistant", "content": "The sky is green."}
  ]
}
```

### 3. Mixed Format (Implicit Prompt)

When `chosen` and `rejected` contain complete conversations, the system automatically extracts the common prompt:

```json
{
  "chosen": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "The sky is blue."}
  ],
  "rejected": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "The sky is green."}
  ]
}
```

This gets automatically processed to:
```json
{
  "prompt": [{"role": "user", "content": "What color is the sky?"}],
  "chosen": [{"role": "assistant", "content": "The sky is blue."}],
  "rejected": [{"role": "assistant", "content": "The sky is green."}]
}
```

## Code Completion Examples

For code completion tasks, the data format is as follows:

### Basic Code Completion

```json
{
  "prompt": "def calculate_sum(a, b):",
  "chosen": "\n    return a + b",
  "rejected": "\n    return a * b"
}
```

### Contextual Code Completion

```json
{
  "prompt": "class Calculator:\n    def __init__(self):\n        self.history = []\n    \n    def add(self, a, b):",
  "chosen": "\n        result = a + b\n        self.history.append(f'{a} + {b} = {result}')\n        return result",
  "rejected": "\n        return a + b"
}
```

### Multi-language Code Completion

```json
{
  "prompt": "// JavaScript function to check if number is prime\nfunction isPrime(num) {",
  "chosen": "\n    if (num <= 1) return false;\n    for (let i = 2; i <= Math.sqrt(num); i++) {\n        if (num % i === 0) return false;\n    }\n    return true;\n}",
  "rejected": "\n    return num > 1;\n}"
}
```

## Data Processing Pipeline

1. **Prompt Extraction**: The system uses `maybe_extract_prompt()` to automatically extract the common prefix from `chosen` and `rejected` as the `prompt`

2. **Chat Template Application**: If a chat template is configured, the system uses `maybe_apply_chat_template()` to process conversational formats

3. **Tokenization**: Convert text to model input format:
   - `prompt_input_ids` / `prompt_attention_mask`
   - `chosen_input_ids` / `chosen_attention_mask`  
   - `rejected_input_ids` / `rejected_attention_mask`

4. **Data Collation**: Use `DPODataCollatorWithPadding` for batching and padding

## Configuration Parameters

You can configure the following parameters to control data processing during training:

- `max_length`: Maximum sequence length (default 512)
- `max_prompt_length`: Maximum prompt length (default 128)
- `max_completion_length`: Maximum completion length for encoder-decoder models (default 128)
- `truncation_mode`: Truncation mode ("keep_start" or "keep_end")

## Dataset Loading Examples

### Loading from Local Disk

```python
from datasets import load_from_disk
from trl import DPOTrainer

# Load dataset
dataset = load_from_disk("path/to/dpo_dataset")

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    args=training_args,
)
```

### Loading from Hub

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("your-dataset-name")

# Data format will be processed automatically
trainer = DPOTrainer(
    model=model,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    args=training_args,
)
```

## Important Notes

1. **Field Consistency**: Ensure all samples contain the required fields
2. **Length Control**: Pay attention to text length to avoid exceeding model limits
3. **Quality Difference**: There should be a clear quality difference between `chosen` and `rejected`
4. **Format Uniformity**: Maintain consistent format within the same dataset
5. **Encoding**: Ensure text uses UTF-8 encoding

## Dataset Validation

You can use the following code to validate dataset format:

```python
def validate_dpo_dataset(dataset):
    """Validate DPO dataset format"""
    required_keys = {"prompt", "chosen", "rejected"}
    
    for i, example in enumerate(dataset):
        # Check required fields
        if not required_keys.issubset(example.keys()):
            print(f"Sample {i} missing required fields: {required_keys - set(example.keys())}")
            
        # Check field types
        for key in required_keys:
            if key in example:
                value = example[key]
                if not isinstance(value, (str, list)):
                    print(f"Sample {i} field {key} has wrong type: {type(value)}")
                    
    print("Dataset validation completed")

# Usage example
validate_dpo_dataset(dataset["train"])
```

By following the above format specifications, you can ensure the correctness and compatibility of DPO training data.