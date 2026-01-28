# EvoMed: Multi-Specialty Medical Diagnosis System

EvoMed is a high-performance medical auxiliary diagnosis system based on an **Evolvable Expert Pool (EEP)** and a **Modular Experience Library (ACE)**. It leverages genetic algorithms to optimize diagnostic prompts for various clinical specialties and utilizes a modular experience library for continuous learning and adaptation.

## 🚀 Key Features

- **Evolvable Expert Pool (EEP)**: Dynamically activates and evolves specialized AI agents based on diagnostic complexity and expert divergence.
- **Modular Experience Library (ACE)**: Captures and updates clinical experiences in a structured way, separating immutable context from iterative hypotheses.
- **Hybrid Retrieval System**: Combines RAG (Retrieval-Augmented Generation) with medical guidelines, structured experiences, and similar cases.
- **Multi-Specialty Coordination**: Implements a multi-step pipeline (Routing -> Rewriting -> Differential Diagnosis -> Aggregation) for comprehensive medical assessment.

## 📁 Repository Structure

### Core Package (`evomed/`)

- **`evomed/`**: The main Python package containing all source code.
  - **`evomem/`**: Memory and experience management.
    - **`generate/`**: Modular Experience Library (ACE) core logic.
    - **`storage/`**: Agentic Memory (A-Mem) storage and retrieval logic.
  - **`evoexperts/`**: Genetic Algorithm (GA) evolution framework for expert prompts.
  - **`data/`**: Data processing and loading modules.
  - **`models/`**: AI models and expert pool management.
  - **`retrieval/`**: Hybrid retrieval engine (RAG, ACE, and Case libraries).
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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
