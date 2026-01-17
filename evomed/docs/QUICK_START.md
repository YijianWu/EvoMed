# 快速开始指南

## 环境准备

### 1. 系统要求
- Python 3.8+
- 8GB+ RAM
- OpenAI API密钥或兼容的LLM API

### 2. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/medical-diagnosis-system.git
cd medical-diagnosis-system

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置API密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，使用兼容API时设置
```

或者在代码中配置：

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## 使用已优化的专家池

### 基础诊断示例

```python
from src.main_diagnosis_pipeline import DiagnosticPipeline, PatientInfo

# 1. 初始化诊断流水线
pipeline = DiagnosticPipeline(
    activation_mode="eep_semantic",
    enable_rag=False
)

# 2. 构建患者信息
patient = PatientInfo(
    patient_id="test_001",
    age=45,
    gender="女",
    chief_complaint="右上腹痛3天",
    present_illness="患者3天前无明显诱因出现右上腹痛...",
    past_history="既往体健",
    physical_exam="右上腹压痛阳性，墨菲氏征阳性",
    lab_results="WBC 12.5×10^9/L",
    diagnosis=""
)

# 3. 执行诊断
result = pipeline.diagnose(patient)
print(result)
```

### 运行示例代码

```bash
# 基础诊断示例
cd examples
python basic_diagnosis_example.py
```

## 训练自定义专家池

### 1. 准备数据

参见 `data/README.md`，准备符合格式的患者数据集。

### 2. 训练单个科室

```python
from src.training.run_specialty_evolution import run_specialty_evolution
from src.main_diagnosis_pipeline import DiagnosticPipeline
import pandas as pd

# 加载数据
df = pd.read_excel("data/guilin_inpatient_extracted_10000.xlsx")

# 筛选验证病例
patients = load_patients_by_specialty(df, "肿瘤科", count=5)

# 初始化流水线
pipeline = DiagnosticPipeline(
    activation_mode="eep_semantic",
    enable_rag=False
)

# 运行训练
experts = run_specialty_evolution(
    specialty="肿瘤科",
    patients=patients,
    pipeline=pipeline,
    generations=10,
    population_size=16,
    top_k_return=2
)
```

### 3. 训练所有科室

```bash
# 完整训练（约4-8小时）
python src/training/run_specialty_evolution.py

# 中断后继续
python src/training/continue_specialty_evolution.py
```

## 批量评估

```bash
# 批量诊断
python src/evaluation/batch_diagnosis.py

# 并发批量诊断（更快）
python src/evaluation/batch_diagnosis_concurrent.py
```

## 启用知识增强

### 1. 构建RAG索引

```python
from rag.rag_build import build_rag_index

# 构建索引
build_rag_index(
    docs_dir="rag/腹痛指南",
    index_dir="rag/rag_index"
)
```

### 2. 使用RAG检索

```python
pipeline = DiagnosticPipeline(
    activation_mode="eep_semantic",
    enable_rag=True,  # 启用RAG
    enable_experience=False,
    enable_case=False
)

result = pipeline.diagnose(patient)
```



