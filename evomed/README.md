# Multi-Specialty Medical Diagnosis System - Based on Evolvable Expert Pool

Multi-Specialty Medical Diagnosis System with Evolvable Expert Pool

> **🎯 Recommended: Quick Experience with Web UI!** 
> 
> ```bash
> ./run_webui.sh
> # Visit http://localhost:8501
> ```
> 
> 📖 See [Web UI Quick Start Guide](QUICK_START_WEBUI.md)

## 📋 Project Overview

This project proposes a multi-specialty medical diagnosis assistant system based on an **Evolvable Expert Pool (EEP)**. By using genetic algorithms to automatically optimize diagnostic prompts for 14 clinical departments, it builds a high-performance diagnostic system containing 28 experts.

### Key Features

- **🧬 Evolvable Expert Pool**: Automatic optimization of diagnostic prompts for each specialty using genetic algorithms.
- **🏥 Multi-Specialty Coverage**: Covers 14 clinical departments, with 2 expert variants per department.
- **🔄 Four-Step Diagnostic Process**: Routing → Information Reconstruction → Expert Diagnosis → Result Aggregation.
- **📚 Knowledge Enhancement**: Integrates RAG retrieval, experience base, and case library.

## 🏗️ System Architecture

```
Diagnostic Flow:
Patient Info Input
    ↓
Step 1: Department Routing (system_step1_route.py)
    ↓
Step 2: Information Reconstruction (system_step2_ir.py)
    ↓
Step 3: Expert Diagnosis (system_step3_diag.py)
    ↓
Step 4: Result Aggregation (system_step4_agg.py)
    ↓
Output Diagnosis Result
```

## 🏥 14 Clinical Departments

1. **Obstetrics and Gynecology**
2. **Gastroenterology**
3. **Pediatrics**
4. **Endocrinology**
5. **Hepatobiliary Surgery**
6. **Orthopedics**
7. **Respiratory Medicine**
8. **Emergency Medicine**
9. **Urology**
10. **General Practice**
11. **Gastrointestinal Surgery**
12. **Cardiothoracic Surgery**
13. **Oncology**
14. **Cardiology**

## 📦 Installation

### 1. Clone Project

```bash
git clone https://github.com/YijianWu/EvoMed.git
cd EvoMed/evomed
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Configure your API key in `src/diagnosis_api.py`:

```python
API_BASE_URL = "https://your-api-endpoint.com/v1"
API_KEY = "your-api-key-here"
MODEL_NAME = "gpt-4o"  # or other models
```

## 🚀 Quick Start

### Method 1: Python API Call

```python
from src.main_diagnosis_pipeline import DiagnosticPipeline, PatientInfo

# Initialize pipeline
pipeline = DiagnosticPipeline(
    activation_mode="eep_semantic",  # Use Evolvable Expert Pool
    enable_rag=True,                 # Enable RAG
    enable_experience=False,
    enable_case=False
)

# Build patient info
patient = PatientInfo(
    patient_id="test_001",
    age=45,
    gender="Female",
    chief_complaint="Abdominal pain for 3 days",
    presentIllness="Patient experienced abdominal pain 3 days ago...",
    # ...
)

# Execute diagnosis
result = pipeline.diagnose(patient)
print(result)
```

### Method 2: Command Line (CLI) Usage

#### Basic Usage

```bash
# Diagnose using patient data file
python src/diagnosis_api.py --patient_file patient.json
```

#### Full Parameters

```bash
python src/diagnosis_api.py \
    --patient_file patient.json \
    --c_file bowel_sound.json \
    --ecg-file ecg.json
```

For file format details, please refer to `DIAGNOSIS_API_DOCS.md`.

### Method 3: Web UI Interface (Recommended)

The simplest way to use, providing a visual Web interface.

#### Start Web UI

```bash
./run_webui.sh
```

Visit: http://localhost:8501

### Method 4: HTTP API Server

#### Start Server

```bash
python src/api_server.py
```

Access API Docs: http://localhost:8000/docs

## 🎓 Training Custom Expert Pool

### Run Full Training

```bash
python src/training/run_specialty_evolution.py
```

## 📁 Directory Structure

```
.
├── src/                          # Source Code
│   ├── system_step1_route.py     # Step 1: Routing
│   ├── system_step2_ir.py        # Step 2: Info Reconstruction
│   ├── system_step3_diag.py      # Step 3: Diagnosis
│   ├── system_step4_agg.py       # Step 4: Aggregation
│   ├── main_diagnosis_pipeline.py # Main Pipeline
│   ├── diagnosis_api.py          # CLI Interface
│   ├── api_server.py            # HTTP API Server
│   └── ...
├── outputs/                      # Outputs
│   └── optimized_expert_pool_28.json  # Optimized 28-Expert Pool
├── data/                         # Data Directory
├── evomem/                       # ACE Framework (Submodule)
├── examples/                     # Examples
├── docs/                         # Documentation
├── requirements.txt              # Dependencies
└── README.md                     # This File
```

## 🔧 Advanced Configuration

### Configure Expert Pool Path

```bash
export EXPERT_POOL_PATH=/path/to/your/expert_pool.json
python src/diagnosis_api.py --patient_file patient.json
```

### Enable RAG

```python
pipeline = DiagnosticPipeline(
    activation_mode="eep_semantic",
    enable_rag=True,
    # ...
)
```
