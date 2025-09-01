#!/usr/bin/env python3
"""
DPO Dataset Format Validation and Example Script

This script demonstrates the DPO dataset format and provides validation functionality.
"""

import json
import sys
from typing import Dict, List, Union, Any


def validate_dpo_dataset(dataset: List[Dict[str, Any]]) -> bool:
    """
    验证 DPO 数据集格式
    Validate DPO dataset format
    
    Args:
        dataset: List of dataset examples
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = {"prompt", "chosen", "rejected"}
    valid = True
    
    print("🔍 Validating DPO dataset format...")
    print(f"📊 Total samples: {len(dataset)}")
    
    for i, example in enumerate(dataset):
        # Check required fields
        missing_keys = required_keys - set(example.keys())
        if missing_keys:
            print(f"❌ Sample {i}: Missing required fields: {missing_keys}")
            valid = False
            continue
            
        # Check field types
        for key in required_keys:
            value = example[key]
            if not isinstance(value, (str, list)):
                print(f"❌ Sample {i}: Field '{key}' has invalid type {type(value)}, expected str or list")
                valid = False
                
        # Additional validation for conversational format
        for key in ["prompt", "chosen", "rejected"]:
            if isinstance(example[key], list):
                for j, msg in enumerate(example[key]):
                    if not isinstance(msg, dict):
                        print(f"❌ Sample {i}: {key}[{j}] should be dict, got {type(msg)}")
                        valid = False
                    elif not all(k in msg for k in ["role", "content"]):
                        print(f"❌ Sample {i}: {key}[{j}] missing 'role' or 'content' field")
                        valid = False
                        
    if valid:
        print("✅ Dataset format validation passed!")
    else:
        print("❌ Dataset format validation failed!")
        
    return valid


def create_example_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """
    创建示例数据集
    Create example datasets showing different DPO formats
    """
    
    # 1. Simple string format for code completion
    code_completion_examples = [
        {
            "prompt": "def fibonacci(n):",
            "chosen": "\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "rejected": "\n    return n * 2"
        },
        {
            "prompt": "class Calculator:\n    def add(self, a, b):",
            "chosen": "\n        return a + b",
            "rejected": "\n        return a - b"
        },
        {
            "prompt": "// Find maximum element in array\nfunction findMax(arr) {",
            "chosen": "\n    return Math.max(...arr);",
            "rejected": "\n    return arr[0];"
        }
    ]
    
    # 2. Conversational format
    conversational_examples = [
        {
            "prompt": [
                {"role": "user", "content": "Write a Python function to reverse a string"}
            ],
            "chosen": [
                {"role": "assistant", "content": "Here's a Python function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\n\nThis uses Python's slice notation to reverse the string efficiently."}
            ],
            "rejected": [
                {"role": "assistant", "content": "def reverse(s):\n    return s.reverse()"}
            ]
        },
        {
            "prompt": [
                {"role": "user", "content": "What's the time complexity of bubble sort?"}
            ],
            "chosen": [
                {"role": "assistant", "content": "The time complexity of bubble sort is:\n- Worst case: O(n²)\n- Average case: O(n²)\n- Best case: O(n) when the array is already sorted\n\nSpace complexity is O(1) as it sorts in-place."}
            ],
            "rejected": [
                {"role": "assistant", "content": "Bubble sort is O(n)."}
            ]
        }
    ]
    
    # 3. Mixed format (implicit prompt)
    mixed_format_examples = [
        {
            "chosen": [
                {"role": "user", "content": "Implement quicksort in Python"},
                {"role": "assistant", "content": "Here's a quicksort implementation:\n\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    \n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    \n    return quicksort(left) + middle + quicksort(right)\n```"}
            ],
            "rejected": [
                {"role": "user", "content": "Implement quicksort in Python"},
                {"role": "assistant", "content": "def quicksort(arr):\n    arr.sort()\n    return arr"}
            ]
        }
    ]
    
    return {
        "code_completion": code_completion_examples,
        "conversational": conversational_examples,
        "mixed_format": mixed_format_examples
    }


def save_example_datasets():
    """保存示例数据集到文件"""
    examples = create_example_datasets()
    
    for format_type, dataset in examples.items():
        filename = f"examples/dpo_dataset_{format_type}.jsonl"
        print(f"💾 Saving {format_type} examples to {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')


def load_dataset_from_file(filepath: str) -> List[Dict[str, Any]]:
    """从JSONL文件加载数据集"""
    dataset = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        print(f"📂 Loaded {len(dataset)} examples from {filepath}")
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error in {filepath}: {e}")
    
    return dataset


def main():
    """主函数"""
    print("🚀 DPO Dataset Format Demo")
    print("=" * 50)
    
    # Create and validate example datasets
    examples = create_example_datasets()
    
    for format_type, dataset in examples.items():
        print(f"\n📝 Format: {format_type.upper()}")
        print("-" * 30)
        
        # Show first example
        if dataset:
            print("Example:")
            print(json.dumps(dataset[0], indent=2, ensure_ascii=False))
            
        # Validate format
        is_valid = validate_dpo_dataset(dataset)
        
        if not is_valid:
            print(f"⚠️  Issues found in {format_type} format!")
    
    # Option to save examples
    if len(sys.argv) > 1 and sys.argv[1] == '--save':
        print("\n💾 Saving example datasets...")
        import os
        os.makedirs('examples', exist_ok=True)
        save_example_datasets()
        print("✅ Example datasets saved!")


if __name__ == "__main__":
    main()