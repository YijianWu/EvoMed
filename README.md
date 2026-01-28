# EvoMed: Multi-Specialty Medical Diagnosis System

EvoMed is a high-performance medical auxiliary diagnosis system based on an **Evolvable Expert Pool (EEP)** and a **Modular Experience Library (ACE)**. It leverages genetic algorithms to optimize diagnostic prompts for various clinical specialties and utilizes a modular experience library for continuous learning and adaptation.

## 🚀 Key Features

- **Evolvable Expert Pool (EEP)**: Dynamically activates and evolves specialized AI agents based on diagnostic complexity and expert divergence.
- **Modular Experience Library (ACE)**: Captures and updates clinical experiences in a structured way, separating immutable context from iterative hypotheses.
- **Hybrid Retrieval System**: Combines RAG (Retrieval-Augmented Generation) with medical guidelines, structured experiences, and similar cases.
- **Multi-Specialty Coordination**: Implements a multi-step pipeline (Routing -> Rewriting -> Differential Diagnosis -> Aggregation) for comprehensive medical assessment.

## 📁 Repository Structure

### Core Package (`ace/`)

- **`ace/`**: The main Python package.
  - **`ace/`**: Modular Experience Library (ACE) core logic.
    - `playbook.py`: Structured clinical experience storage.
    - `retrieval.py`: Semantic and modular retrievers.
    - `roles.py`: AI agent roles (Generator, Reflector, Curator).
  - **`data/`**: Data processing and loading modules.
    - `loader.py`: Case parsing and batch loading.
    - `concurrent_loader.py`: High-concurrency diagnostic processing.
  - **`models/`**: AI models and expert pool management.
    - `expert_pool.py`: EEP (Evolvable Expert Pool) management.
  - **`retrieval/`**: Hybrid retrieval engine.
    - `knowledge.py`: Unified interface for RAG, ACE, and Case libraries.
    - `hybrid.py`: BM25 + Vector hybrid search implementation.
  - **`prompt/`**: System prompt templates for each diagnostic step.
  - `diagnosis.py`: Core multi-step diagnostic pipeline.
  - `pipeline.py`: Simplified API interface.
  - `trainer.py`: Genetic Algorithm evolution framework for expert prompts.

### Entry Point Scripts

- `run_server.py`: FastAPI server for HTTP API.
- `run_webui.py`: Streamlit-based graphical user interface.
- `run_evo.py`: Modular evolution entry point.
- `run_rag_build.py`: RAG index construction script.
- `run_all.sh`: Automation script for full batch evolution.

### Support Directories

- **`config/`**: System and model configuration files (e.g., DeepSpeed).
- **`outputs/`**: Generated reports and optimized expert pools.
- **`data/`**: Raw and processed medical record data files.
- **`docs/`**: Project documentation and references.
- **`exp/`**: Experimental features and external integrations (e.g., A-Mem system).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
