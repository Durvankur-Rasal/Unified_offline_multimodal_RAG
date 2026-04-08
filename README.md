# MediRAG: Privacy-Preserving Agentic Healthcare Assistant 🩺🔒

A Unified Architecture for Privacy-Preserving Agentic RAG on Local Infrastructure

MediRAG is a 100% offline, locally hosted Artificial Intelligence assistant designed specifically for clinical environments. It resolves the critical privacy and hallucination issues of cloud-based Large Language Models (LLMs) by combining Differential Privacy, Retrieval-Augmented Generation (DP-RAG), and Agentic State Machines.

This system guarantees absolute data sovereignty, ensuring sensitive Electronic Health Records (EHR) never leave the local hospital network, while running efficiently on standard consumer hardware (8GB RAM).

## ✨ Key Features

- **100% Offline Processing:** All text embedding, vector storage, intent routing, and natural language generation happen locally. Zero internet dependencies or external APIs.
- **Differential Privacy (DP-RAG):** Mathematical Laplacian noise ($\epsilon=1.2$) is injected into patient data embeddings before being stored in the FAISS vector database, preventing reverse-engineering attacks.
- **Agentic Routing (LangGraph):** The AI doesn't just read text; it thinks. The LangGraph state machine analyzes the clinical query to determine if it should search medical records OR use a deterministic Python calculator (preventing math hallucinations).
- **Hardware-Aware AI:** Powered by the Ollama framework and Microsoft's highly optimized Phi-3 (3.8B) model, designed to run smoothly on laptops with limited memory (8GB RAM) and standard GPUs.
- **Immunity to Hallucination:** Strict prompt engineering and LCEL (LangChain Expression Language) pipelines force the AI to answer only from the retrieved local context or explicitly refuse.

## 🏗️ System Architecture

The project is structured into five distinct, tightly integrated layers:

- **Presentation Layer:** React Frontend Dashboard.
- **Orchestration Layer:** FastAPI backend gateway routing JSON payloads.
- **Agentic Layer (LangGraph):** The semantic router that classifies user intent (search vs calculate).
- **Tool Environment:**
  - **DP-RAG Search Tool:** Encrypted FAISS vector store.
  - **Clinical Calculator:** Deterministic Python logic for BMI, dosages, etc.
- **Inference Engine:** Local Ollama service hosting the Phi-3 Small Language Model.

## 💻 Hardware Prerequisites

- **OS:** Windows 10/11, macOS, or Linux
- **RAM:** Minimum 8GB System RAM (No expensive cloud GPU required)
- **GPU:** NVIDIA GTX/RTX recommended (Ollama will automatically detect and offload to it)

## 🚀 Installation & Setup

### 1. Install the Local AI Engine (Ollama)

Instead of fighting with C++ compilers, this project uses Ollama as a standalone background service.

- Download and install Ollama.
- Open a terminal and download the Phi-3 model:

```bash
ollama pull phi3
```

### 2. Create the Python Environment

It is highly recommended to use Python 3.11 for AI library compatibility.

```bash
conda create -n major_project python=3.11 -y
conda activate major_project
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn langchain langchain-community langchain-core sentence-transformers faiss-cpu pydantic langgraph
```

## 📂 Data Ingestion (Building the DP-RAG Database)

Before the AI can answer questions, you must securely load your medical data into the encrypted FAISS index.

- Place your mock Electronic Health Records (EHR) as `.txt` files inside the `source_documents/` folder.
- Example: `source_documents/EHR_Rajesh_Sharma.txt`

Run the ingestion script to chunk the text, apply Differential Privacy ($\epsilon=1.2$), and save the vectors:

```bash
python ingest.py
```

Verify that the `faissindex/` folder has been generated in your project root.

## 🏃‍♂️ Running the Application

### Start the FastAPI Server

With your `major_project` conda environment active, start the orchestrator:

```bash
python api.py
```

The server will start on `http://localhost:8000`.

### Testing the Endpoints

You can test the agent's routing capabilities using standard curl commands or by connecting your React frontend to the `/chat` endpoint.

#### Test 1: The Search Route (DP-RAG)

```bash
curl -X 'POST' 'http://localhost:8000/chat' \
-H 'Content-Type: application/json' \
-d '{"query": "What is Rajesh Sharma'\''s chief complaint?"}'
```

Expected Output: The agent classifies the intent as search, retrieves the encrypted data, and outputs the symptoms alongside FAISS citations.

#### Test 2: The Math Route (Clinical Calculator)

```bash
curl -X 'POST' 'http://localhost:8000/chat' \
-H 'Content-Type: application/json' \
-d '{"query": "Calculate the BMI for a patient who weighs 80 kg and is 1.75 meters tall."}'
```

Expected Output: The agent classifies the intent as calculate, bypasses the FAISS database completely, and computes the exact BMI using Python logic.

## 📁 Folder Structure

```plaintext
multimodal-offline-rag/
│
├── api.py                     # FastAPI server and endpoint routing
├── ingest.py                  # Script to build the FAISS vector database
├── faissindex/                # Encrypted local vector storage (generated)
├── source_documents/          # Raw patient .txt files
│
└── src/
    ├── dp_embeddings.py       # Custom HuggingFace Wrapper with Laplacian Noise
    ├── agentic_pipeline.py    # LangGraph State Machine (Router & Tools)
    └── rag_pipeline.py        # (Legacy) Standard LCEL RAG pipeline
```

## 🔮 Future Work

While the current architecture securely processes text and deterministic mathematics, future iterations will implement true Multimodal capabilities. By integrating local Vision-Language Models (VLMs), the agent will be able to securely analyze visual diagnostics, such as X-rays, MRI scans, and ECGs, alongside traditional written health records, maintaining the 100% offline, privacy-first guarantee.
