# EvoMed: Multi-Specialty Medical Diagnosis System

EvoMed is a high-performance medical auxiliary diagnosis system based on an **Evolvable Expert Pool (EEP)** and a **Modular Experience Library (ACE)**. It leverages genetic algorithms to optimize diagnostic prompts for various clinical specialties and utilizes a modular experience library for continuous learning and adaptation.

> **🎯 Quick Start with Web UI!**
>
> ```bash
> ./run_webui.sh
> # Access http://localhost:8501
> ```

## 📋 Project Overview

EvoMed integrates advanced AI techniques to provide comprehensive diagnostic support:

- **🧬 Evolvable Expert Pool (EEP)**: Automatically optimizes diagnostic prompts for 16 clinical departments using genetic algorithms.
- **🧠 Modular Experience Library (ACE)**: A modular system for retrieving, reflecting, and evolving clinical experiences to improve diagnostic accuracy over time.
- **🏥 Multi-Specialty Coverage**: Covers 16 clinical departments with a pool of 64 optimized experts.
- **🔄 Four-Step Diagnostic Pipeline**: 
    1. **Routing**: Department-level triage and planning.
    2. **Information Reconstruction**: Specialty-specific medical history rewriting.
    3. **Differential Diagnosis**: Expert-level analysis with knowledge augmentation.
    4. **Aggregation**: Multi-expert consensus and final decision making.
- **📚 Knowledge Augmentation**: Integrated RAG (Medical Guidelines), Experience Library (A-Mem), and Case Library (ACE).

## 🏗️ System Architecture

```
Diagnostic Workflow:
Input Patient Info
    ↓
Step 1: Specialty Routing (src/system_step1_route.py)
    ↓
Step 2: Info Reconstruction (src/system_step2_ir.py)
    ↓
Step 3: Expert Diagnosis (src/system_step3_diag.py) + Knowledge Retrieval
    ↓
Step 4: Consensus Aggregation (src/system_step4_agg.py)
    ↓
Final Diagnostic Report
```

## 🏥 Clinical Departments Covered

1. **Obstetrics & Gynecology**
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
15. **Rheumatology**
16. **Neurology**

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/EvoMed.git
cd EvoMed
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Configure API Keys

Edit `src/main_diagnosis_pipeline.py` or `src/diagnosis_api.py` to set your API credentials:

```python
API_BASE_URL = "https://your-api-endpoint.com/v1"
API_KEY = "your-api-key-here"
MODEL_NAME = "gpt-4o"
```

## 🚀 Usage

### Web UI (Recommended)

The easiest way to experience EvoMed is through the Streamlit-based Web UI.

```bash
./run_webui.sh
```

### Command Line Interface (CLI)

Perform diagnosis on a patient data file:

```bash
python src/main.py --mode evolved_pool --top_k 8
```

### Experience Library (ACE) Evolution

To run the experience library evolution process:

```bash
python scripts/run_modular_evolution.py
```

## 📁 Repository Structure

```
.
├── ace/                # Modular Experience Library (ACE) core logic
├── src/                # Core diagnostic pipeline and expert pool management
├── scripts/            # Runnable scripts for evolution and testing
├── outputs/            # Optimized expert pools and diagnostic reports
│   └── moa_optimized_expert_pool_64.json  # Current best 64-expert pool
├── data/               # Datasets and patient records
├── rag/                # RAG knowledge base and index
├── examples/           # Usage examples
├── docs/               # Technical documentation
├── tests/              # Unit and integration tests
├── run_webui.sh        # Web UI startup script
└── requirements.txt    # Python dependencies
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

