# EvoMed: Multi-Specialty Medical Diagnosis System

EvoMed is a high-performance medical auxiliary diagnosis system based on an **Evolvable Expert Pool (EEP)** and a **Modular Experience Library (ACE)**. It leverages genetic algorithms to optimize diagnostic prompts for various clinical specialties and utilizes a modular experience library for continuous learning and adaptation.

## 🚀 Key Features

- **Evolvable Expert Pool (EEP)**: Dynamically activates and evolves specialized AI agents based on diagnostic complexity and expert divergence.
- **Modular Experience Library (ACE)**: Captures and updates clinical experiences in a structured way, separating immutable context from iterative hypotheses.
- **Hybrid Retrieval System**: Combines RAG (Retrieval-Augmented Generation) with medical guidelines, structured experiences, and similar cases.
- **Multi-Specialty Coordination**: Implements a multi-step pipeline (Routing -> Rewriting -> Differential Diagnosis -> Aggregation) for comprehensive medical assessment.

## 📁 Repository Structure

### Core Directories

- **`ace/`**: Contains the core logic for the Modular Experience Library (ACE).
  - `playbook.py`: Defines structured clinical experience storage.
  - `retrieval.py`: Implements semantic and modular retrievers using SentenceTransformers and FAISS.
  - `roles.py`: Defines AI roles (Generator, Reflector, Curator) within the ACE framework.
- **`src/`**: Core diagnostic pipeline and system management.
  - `main_diagnosis_pipeline.py`: Main entry point for the medical diagnosis workflow.
  - `expert_pool.py`: Management of the expert agent collection.
  - `diagnosis_api.py`: Simplified API interface for diagnostic services.
  - `api_server.py`: FastAPI server for exposing diagnostic functions over HTTP.
  - `knowledge_retriever.py`: Unified interface for retrieving from RAG, ACE, and Case libraries.
  - `hybrid_retriever.py`: Hybrid search implementation (BM25 + Vector) for medical guidelines.
  - **`training/`**: Scripts for evolving expert pool prompts using Genetic Algorithms.
  - **`evaluation/`**: Scripts for batch diagnosis and performance assessment.
- **`rag/`**: RAG-related modules and index building.
  - `rag_build.py`: Script to build the FAISS index from medical guideline documents.
- **`scripts/`**: Automation scripts for modular evolution and testing.
- **`outputs/`**: Storage for optimized expert pools and generated diagnostic reports.

### Key Files

- `run_webui.sh`: Startup script for the Streamlit-based Web UI.
- `requirements.txt`: Python package dependencies.
- `README.md`: This project documentation.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
