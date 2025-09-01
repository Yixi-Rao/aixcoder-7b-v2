# DPO 数据集格式说明

DPO (Direct Preference Optimization) 是一种直接优化人类偏好的训练方法。本文档详细说明了在本项目中使用的 DPO 数据集的格式要求。

## 核心字段

DPO 数据集必须包含以下三个核心字段：

- **`prompt`**: 输入提示，可以是字符串或对话格式
- **`chosen`**: 首选的回答/补全内容  
- **`rejected`**: 被拒绝的回答/补全内容

## 数据格式类型

### 1. 简单字符串格式

最基本的格式，所有字段都是简单的字符串：

```json
{
  "prompt": "请实现一个计算斐波那契数列的函数",
  "chosen": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "rejected": "def fib(n):\n    return n + n - 1"
}
```

### 2. 对话格式

适用于聊天场景，每个字段都是消息列表：

```json
{
  "prompt": [
    {"role": "user", "content": "什么颜色的天空?"}
  ],
  "chosen": [
    {"role": "assistant", "content": "天空是蓝色的。"}
  ],
  "rejected": [
    {"role": "assistant", "content": "天空是绿色的。"}
  ]
}
```

### 3. 混合格式（隐式提示）

当 `chosen` 和 `rejected` 中包含完整对话时，系统会自动提取共同的提示部分：

```json
{
  "chosen": [
    {"role": "user", "content": "什么颜色的天空?"},
    {"role": "assistant", "content": "天空是蓝色的。"}
  ],
  "rejected": [
    {"role": "user", "content": "什么颜色的天空?"},
    {"role": "assistant", "content": "天空是绿色的。"}
  ]
}
```

系统会自动处理为：
```json
{
  "prompt": [{"role": "user", "content": "什么颜色的天空?"}],
  "chosen": [{"role": "assistant", "content": "天空是蓝色的。"}],
  "rejected": [{"role": "assistant", "content": "天空是绿色的。"}]
}
```

## 代码补全场景示例

对于代码补全任务，数据格式如下：

### 基础代码补全

```json
{
  "prompt": "def calculate_sum(a, b):",
  "chosen": "\n    return a + b",
  "rejected": "\n    return a * b"
}
```

### 上下文代码补全

```json
{
  "prompt": "class Calculator:\n    def __init__(self):\n        self.history = []\n    \n    def add(self, a, b):",
  "chosen": "\n        result = a + b\n        self.history.append(f'{a} + {b} = {result}')\n        return result",
  "rejected": "\n        return a + b"
}
```

### 多语言代码补全

```json
{
  "prompt": "// JavaScript function to check if number is prime\nfunction isPrime(num) {",
  "chosen": "\n    if (num <= 1) return false;\n    for (let i = 2; i <= Math.sqrt(num); i++) {\n        if (num % i === 0) return false;\n    }\n    return true;\n}",
  "rejected": "\n    return num > 1;\n}"
}
```

## 数据处理流程

1. **提示提取**: 系统使用 `maybe_extract_prompt()` 函数自动从 `chosen` 和 `rejected` 中提取共同前缀作为 `prompt`

2. **聊天模板应用**: 如果设置了聊天模板，系统会使用 `maybe_apply_chat_template()` 处理对话格式

3. **分词处理**: 将文本转换为模型输入格式：
   - `prompt_input_ids` / `prompt_attention_mask`
   - `chosen_input_ids` / `chosen_attention_mask`  
   - `rejected_input_ids` / `rejected_attention_mask`

4. **数据整理**: 使用 `DPODataCollatorWithPadding` 进行批处理和填充

## 配置参数

训练时可以配置以下参数控制数据处理：

- `max_length`: 最大序列长度（默认 512）
- `max_prompt_length`: 最大提示长度（默认 128）
- `max_completion_length`: 最大补全长度（编码器-解码器模型，默认 128）
- `truncation_mode`: 截断模式（"keep_start" 或 "keep_end"）

## 数据集加载示例

### 从本地加载

```python
from datasets import load_from_disk
from trl import DPOTrainer

# 加载数据集
dataset = load_from_disk("path/to/dpo_dataset")

# 初始化训练器
trainer = DPOTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    args=training_args,
)
```

### 从 Hub 加载

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("your-dataset-name")

# 数据会自动处理格式
trainer = DPOTrainer(
    model=model,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    args=training_args,
)
```

## 注意事项

1. **字段一致性**: 确保所有样本都包含必需的字段
2. **长度控制**: 注意控制文本长度，避免超出模型限制
3. **质量差异**: `chosen` 和 `rejected` 之间应有明显的质量差异
4. **格式统一**: 同一数据集中保持格式的一致性
5. **编码格式**: 确保文本使用 UTF-8 编码

## 验证数据集

可以使用以下代码验证数据集格式：

```python
def validate_dpo_dataset(dataset):
    """验证 DPO 数据集格式"""
    required_keys = {"prompt", "chosen", "rejected"}
    
    for i, example in enumerate(dataset):
        # 检查必需字段
        if not required_keys.issubset(example.keys()):
            print(f"样本 {i} 缺少必需字段: {required_keys - set(example.keys())}")
            
        # 检查字段类型
        for key in required_keys:
            if key in example:
                value = example[key]
                if not isinstance(value, (str, list)):
                    print(f"样本 {i} 的字段 {key} 类型错误: {type(value)}")
                    
    print("数据集验证完成")

# 使用示例
validate_dpo_dataset(dataset["train"])
```

通过遵循以上格式规范，可以确保 DPO 训练数据的正确性和兼容性。