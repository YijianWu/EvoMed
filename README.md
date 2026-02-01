# EvoMed 

Accurate diagnosis is crucial in healthcare. Here, we introduce **EvoMed**, a medical auxiliary diagnostic system focused on multi-specialty collaboration and sustainable evolution. 

The system combines an **Evolvable Expert Pool** and a **Modular Experience Library**, utilizing genetic algorithms and hybrid retrieval techniques to dynamically optimize diagnostic strategies, ensuring evidence-based precision support that adapts to changes in clinical scenarios. 

This repository contains the code for data processing, agent development, and evaluation used in our research. 


## 📁 Repository Structure

### Core Package (`evomed/`)

- **`evomed/`**: The main Python package containing all source code.
  - **`evomem/`**: Memory and experience management.
    - **`engine/`**: Modular Experience Library (Engine) core logic.
    - **`repository/`**: Knowledge Repository storage and retrieval logic.
  - **`evoexperts/`**: Genetic Algorithm (GA) evolution framework for expert prompts.
  - **`data/`**: Data processing and loading modules.
  - **`models/`**: AI models and expert pool management.
  - **`retrieval/`**: Hybrid retrieval engine (RAG, Engine, and Case libraries).
  - **`prompt/`**: System prompt templates for each diagnostic step.
  - `diagnosis.py`: Core multi-step diagnostic pipeline.
  - `pipeline.py`: Simplified API interface.
  - `trainer.py`: GA training orchestration.

### Entry Point Scripts

- `run_server.py`: FastAPI server for HTTP API.
- `run_webui.py`: Streamlit-based graphical user interface.
- `run_evo.py`: Modular evolution entry point.
- `run_rag_build.py`: RAG index construction script.
- `run_all.sh`: Automation script for full batch evolution.

### Support Directories

- **`config/`**: System and model configuration files (e.g., DeepSpeed).
- **`outputs/`**: Generated reports and optimized expert pools.
- **`docs/`**: Project documentation and references.


### Python API Example

```python
from evomed.diagnosis import DiagnosticPipeline, PatientInfo

# 1. Initialize the multi-specialty diagnostic pipeline
pipeline = DiagnosticPipeline(
    activation_mode="evolved_pool",
    evolved_pool_path="outputs/moa_optimized_expert_pool_64.json"
)

# 2. Prepare patient clinical data
patient = PatientInfo(
    patient_id="DEMO_001",
    gender="Male",
    age=52,
    department="Internal Medicine",
    chief_complaint="Abdominal pain with nausea for 2 days",
    history_of_present_illness="Sudden onset of upper abdominal colic...",
    past_history="Hypertension for 5 years",
    personal_history="No special history",
    physical_examination="RUQ tenderness (+), Murphy sign (+)",
    labs="WBC: 11.2×10^9/L",
    imaging="Gallbladder wall thickening on US",
    main_diagnosis="",
    main_diagnosis_icd=""
)

# 3. Execute the four-step diagnostic process
results = pipeline.run_pipeline(patient, top_k=5)

# 4. Output the final consensus diagnosis
print("Integrated Diagnosis Output:\n", results['steps']['step4']['output'])
```

### Command Line Interface

You can also run the system via the provided scripts:

```bash
# Start the Web UI (Streamlit)
streamlit run run_webui.py

# Start the REST API Server (FastAPI)
python run_server.py --host 0.0.0.0 --port 8000
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
