# Build AI Agents: Hands-on RAG Applications

This repository contains practical implementations of **Retrieval-Augmented Generation (RAG)** systems using local LLMs (Ollama Llama 3.2:3b), local embedding models (`mxbai-embed-large`), and vector databases (Chroma).

---

## Branches Overview

* **`main`**: Restaurant Reviews RAG Application (CSV Ingestion & Review Question-Answering).
* **`feature/college-results-rag`**: College Examination Records PDF QA Application (102-page PDF, 200 CSE Engineering Students, 1st to 8th Semester transcripts, exact calculation engine, source citations, and guardrails).

---

## College Exam Results PDF QA System (`feature/college-results-rag`)

### Features
1. **102-Page Examination Gazette PDF**: Complete transcripts for 200 Computer Science & Engineering students across Semesters 1 to 8.
2. **Deterministic Analytics Engine**: Exact calculations for class average CGPA, semester toppers, individual student scorecards, and side-by-side student comparisons.
3. **Semantic Vector Search (Chroma)**: Fast semantic retrieval of curriculum, syllabus, and academic grading policies with local Chroma persistence.
4. **Verifiable Source Citations**: Every answer cites the exact PDF page numbers (`[Source: college_exam_results.pdf, Page X]`).
5. **Hallucination Defense**: Refuses out-of-scope questions without supporting context.
6. **Real-time Streaming**: Tokens stream live in the terminal for instant responses.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed with the following models:
  ```bash
  ollama pull llama3.2:3b
  ollama pull mxbai-embed-large
  ```

### 2. Setup Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install langchain langchain-chroma langchain-ollama langchain-text-splitters pypdf reportlab pandas
```

### 3. Running the College Results Assistant
Switch to the feature branch and run:
```powershell
git checkout feature/college-results-rag
python main.py
```

### 4. Running the Restaurant Reviews Assistant
Switch to the main branch and run:
```powershell
git checkout main
python main.py
```
