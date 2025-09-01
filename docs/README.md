# docs/README.md

## Documentation Overview

This directory contains documentation for the aiXcoder-colt project, with detailed specifications for data formats and validation tools.

### DPO Dataset Format Documentation

- **[中文版本 - DPO 数据集格式说明](DPO_DATASET_FORMAT.md)**: 详细的中文版本 DPO 数据集格式规范，包含所有支持的格式类型和示例。

- **[English Version - DPO Dataset Format Specification](DPO_DATASET_FORMAT_EN.md)**: Comprehensive English documentation covering all supported DPO dataset formats with examples.

### Tools and Utilities

- **[DPO Format Validation Script](validate_dpo_format.py)**: Python script to validate DPO dataset format and generate example datasets.

### Quick Start

To validate your DPO dataset format, run:

```bash
python docs/validate_dpo_format.py
```

To generate example datasets:

```bash
python docs/validate_dpo_format.py --save
```

### Supported Formats

The DPO implementation supports three main dataset formats:

1. **Simple String Format**: Basic prompt-response pairs as strings
2. **Conversational Format**: Structured chat messages with roles
3. **Mixed Format**: Full conversations where prompts are automatically extracted

For detailed specifications and examples, refer to the format documentation linked above.