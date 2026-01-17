# Quick Start Guide

## 快速开始指南

### 1. 环境已设置完成 ✅

项目结构已创建，所有依赖已安装。

### 2. 运行示例

#### 基本使用：
```bash
cd /Users/yue/Documents/code/nlpproject/multimodal-mt-eval
/Users/yue/Documents/code/nlpproject/.venv/bin/python examples/basic_usage.py
```

#### 批量评估：
```bash
/Users/yue/Documents/code/nlpproject/.venv/bin/python examples/batch_evaluation.py
```

### 3. 运行测试

```bash
# 安装测试依赖
/Users/yue/Documents/code/nlpproject/.venv/bin/python -m pip install pytest pytest-cov

# 运行测试
cd /Users/yue/Documents/code/nlpproject/multimodal-mt-eval
/Users/yue/Documents/code/nlpproject/.venv/bin/python -m pytest tests/ -v
```

### 4. 使用项目

在Python代码中使用：

```python
from multimodal_mt_eval import MultimodalMTEvaluator

# 创建评估器
evaluator = MultimodalMTEvaluator(
    metrics=["bleu", "bert_score"],
    device="cpu"
)

# 评估翻译
predictions = ["Your translation here"]
references = ["Reference translation"]
results = evaluator.evaluate(predictions, references)
print(results)
```

### 5. 下一步

- 在 `src/multimodal_mt_eval/` 中添加新的评估指标
- 在 `examples/` 中创建更多使用示例
- 扩展 `data_loader.py` 以支持更多数据格式
- 实现多模态评估指标（CLIP等）

### 6. Git 工作流

```bash
# 查看状态
cd /Users/yue/Documents/code/nlpproject/multimodal-mt-eval
git status

# 添加文件
git add .

# 提交更改
git commit -m "Initial project setup with evaluation framework"

# 推送到GitHub
git push origin main
```

### 7. 项目文件说明

- `src/multimodal_mt_eval/`: 核心代码库
  - `evaluator.py`: 主评估器
  - `metrics.py`: 评估指标实现
  - `data_loader.py`: 数据加载工具
- `examples/`: 使用示例
- `tests/`: 单元测试
- `requirements.txt`: 项目依赖
- `setup.py`: 安装配置
- `config.yaml`: 配置文件

### 8. 常用命令

```bash
# 激活虚拟环境（如果需要）
source /Users/yue/Documents/code/nlpproject/.venv/bin/activate

# 安装新的依赖
/Users/yue/Documents/code/nlpproject/.venv/bin/python -m pip install <package_name>

# 重新安装项目（如果修改了setup.py）
cd /Users/yue/Documents/code/nlpproject/multimodal-mt-eval
/Users/yue/Documents/code/nlpproject/.venv/bin/python -m pip install -e .
```
