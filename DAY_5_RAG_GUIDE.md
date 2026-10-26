# Day 5: Retrieval-Augmented Generation (RAG) & College Exam Results Project

Comprehensive reference guide, architectural breakdown, and hands-on project documentation.

---

## 1. What is Retrieval-Augmented Generation (RAG)?

### The Core Problem of Standalone LLMs
Pre-trained Large Language Models (LLMs) such as `llama3.2` or `gpt-4` are powerful, but they have major enterprise limitations:
1. **Knowledge Cutoff Date**: The model knows only what was in its training set up to its release date.
2. **Zero Access to Private Data**: An LLM has no prior knowledge of your company's proprietary documents, internal database records, or student exam marks.
3. **Hallucination Risk**: When asked about specific unlearned facts (e.g., *"What was Aarav Sharma's 4th semester SGPA?"*), standalone LLMs will invent plausible-sounding but false answers.

### The RAG Solution
**RAG (Retrieval-Augmented Generation)** connects an LLM to an external dynamic knowledge base. Instead of relying solely on parametric memory (weights), RAG uses a three-step lifecycle:

```
[ User Query ]
      │
      ├──► 1. RETRIEVE: Convert query to vector -> Search Vector DB (Chroma)
      │                 Fetch Top-K relevant document chunks + Page metadata
      │
      ├──► 2. CONSTRUCT CONTEXT: Inject retrieved chunks & calculations into prompt
      │
      └──► 3. GENERATE: LLM generates grounded answer strictly from context + source citations!
```

---

## 2. Document Ingestion, Parsing, and Chunking

### A. Document Ingestion
Raw unstructured files (PDF, DOCX, CSV, HTML) must be ingested and converted into structured text objects with metadata:
- In our project, `pypdf.PdfReader` ingests `college_exam_results.pdf` (102 pages).
- Page numbers, file names, and student roll numbers (USNs) are attached as **metadata** to each document.

### B. Text Chunking Strategies
Why chunk text?
- **Context Window Limits**: Passing an entire 100-page document into the LLM exceeds context limits and increases latency and cost.
- **Semantic Specificity**: Smaller, focused passages produce sharper embedding vectors that match specific user questions.

| Chunking Strategy | Mechanism | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Fixed-Size (Naive)** | Cuts every $N$ characters | Simple | Breaks sentences and table rows mid-word |
| **Sentence-Based** | Splits on punctuation (`.`, `!`, `?`) | Respects grammar | Can produce uneven chunk sizes |
| **Recursive Character Splitter** (Used in our project) | Tries paragraph (`\n\n`), then line (`\n`), then space (` `) | Preserves semantic blocks & tables | Requires tuning chunk size and overlap |

### C. The Sliding Window (Chunk Overlap)
- **Chunk Size**: The target size of each text segment (e.g., 1,100 characters).
- **Chunk Overlap**: The number of overlapping characters between adjacent chunks (e.g., 150 characters).
- **Why Overlap Matters**: Prevents critical context that spans across chunk boundaries from being lost.

---

## 3. Embeddings and Semantic Search

### What is an Embedding?
An embedding model (like `mxbai-embed-large`) transforms words, sentences, or paragraphs into a high-dimensional vector of numbers (e.g., 1,024 dimensions):

$$\text{Text} \xrightarrow{\text{Embedding Model}} [0.024, -0.187, 0.541, \dots, -0.093]$$

Words and sentences with similar meanings map to points located near each other in this high-dimensional mathematical space.

### Keyword (Lexical) Search vs. Semantic Search
- **Keyword Search (BM25 / TF-IDF)**: Matches exact words. If the query is *"highest scoring student"* but the document says *"Rank 1 merit candidate"*, keyword search can miss it.
- **Semantic Search (Vector Embeddings)**: Understands synonyms, intent, and concepts. It connects *"top scorer"*, *"academic topper"*, and *"first rank candidate"*.

### Similarity Metrics
To determine which chunks are closest to a user query:
1. **Cosine Similarity**: Measures the cosine of the angle between two vectors:
   $$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
   Values range from -1 to 1 (1 means identical semantic orientation).
2. **Euclidean Distance ($L_2$)**: Measures the straight-line distance between two points:
   $$d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum (A_i - B_i)^2}$$

---

## 4. Vector Databases: FAISS vs. Chroma

A Vector Database indexes and stores high-dimensional vectors for fast approximate nearest neighbor (ANN) searches.

| Feature | Chroma (`langchain-chroma`) | FAISS (`faiss-cpu`) |
| :--- | :--- | :--- |
| **Developer** | Chroma Inc. (Open Source) | Meta AI Research |
| **Storage & Persistence** | Built-in SQLite + Parquet on disk | In-memory index; manual serialization |
| **Metadata Filtering** | Native SQL-like metadata filtering | Requires custom external ID mapping |
| **Best Used For** | Full-featured RAG applications with documents, metadata, and persistence | Massive-scale (millions of vectors) pure similarity search |

In this project, we use **Chroma** for local persistence in `./chroma_college_db`.

---

## 5. RAG vs. Fine-Tuning: Architectural Trade-Offs

| Decision Factor | Retrieval-Augmented Generation (RAG) | Fine-Tuning |
| :--- | :--- | :--- |
| **Primary Goal** | Supplying factual, dynamic knowledge | Modifying model behavior, style, or syntax |
| **Updating Information** | Instant (insert/delete chunks in Vector DB) | Slow & Expensive (requires dataset retraining) |
| **Hallucination** | Very Low (answers grounded in provided text) | Moderate to High (can still generate false facts) |
| **Auditability & Citations** | Exact source citations (file name, page number) | Black box (cannot cite source of a specific weight) |
| **Compute Cost** | Low (standard CPU or modest local GPU) | High (requires high-VRAM GPUs for training) |
| **Rule of Thumb** | Use RAG for **what the model knows** | Use Fine-Tuning for **how the model speaks** |

---

## 6. Source Citations & Handling Questions Without Supporting Context

Two critical requirements of enterprise-grade RAG systems:

### 1. Verifiable Source Citations
The user must know exactly where an answer came from:
- In our implementation, each retrieved chunk is tagged with its page number:
  ```
  [Source: college_exam_results.pdf, Page 12]
  ```
- The prompt instructs the LLM to append these exact source citations to every answer.

### 2. Handling Out-of-Context Questions (Guardrails)
When an LLM is asked a question whose answer does not exist in the retrieved documents (e.g. *"What is the cafeteria lunch menu?"* or an unlisted student):
- **Without Guardrails**: The LLM hallucinates an answer.
- **With RAG Guardrails**: The system prompt enforces:
  > *"If the user's question cannot be answered from the provided records, explicitly state: 'I do not have supporting context in the official college examination records to answer this question.'"*

---

## 7. Multi-Turn Conversational RAG & Memory ($k \ge 6$)

In real-world chat interfaces, users rarely ask standalone, isolated questions. Instead, they ask follow-up questions containing pronouns and implicit references:
- Turn 1: *"Compare student 1RV21CS028 and 1RV21CS007"*
- Turn 2: *"What are their respective names?"* (implicit pronoun reference to Turn 1 candidates)
- Turn 3: *"Who scored higher in semester 5?"*

### Core Architectural Components:
1. **Conversation History Buffer ($k \ge 6$)**:
   - The application retains the last $k$ conversation turns (e.g. 6 user questions + 6 assistant answers).
   - Formatted into readable turns (`User:` / `Assistant:`) and injected directly into the LLM prompt.
2. **Context-Aware Entity & Pronoun Resolution**:
   - When a user query lacks an explicit USN or identifier but contains pronouns (*they*, *their*, *both*, *he*, *she*), the router extracts the most recently discussed entities from the conversation history buffer.
   - Authoritative records for those entities are retrieved and supplied to the context.
3. **Disambiguation of Identical Names**:
   - In large academic cohorts, multiple students may share identical names (e.g., two students named *Divya Bhat*).
   - The system pairs names with unique identifiers (USN: `1RV21CS028` vs `1RV21CS007`) and explicit disambiguation alerts to eliminate LLM confusion.

---

## 8. The College Examination RAG Project Overview

### Project Files:
1. `generate_college_data.py`:
   - Synthesizes 200 CSE engineering student records across 8 full semesters.
   - Computes marks (0-100), grades (O, A+, A, B+, B, C, F), grade points, credits, semester SGPAs, and cumulative CGPA.
   - Compiles a 102-page official academic gazette PDF (`college_exam_results.pdf`) using ReportLab.
   - Emits structured JSON (`students_results.json`) for deterministic mathematical queries.
2. `analytics.py`:
   - Exact mathematical calculation engine for top scorers, rank lists, batch averages, semester averages, pass percentages, student scorecards, and comparisons.
3. `vector.py`:
   - Parses `college_exam_results.pdf` using PyPDF.
   - Chunks text with `RecursiveCharacterTextSplitter`.
   - Embeds with local `mxbai-embed-large` and stores in `Chroma` (`./chroma_college_db`).
   - Exposes `retriever` and `retrieve_with_sources()` with citations.
4. `main.py`:
   - Terminal interactive assistant powered by `llama3.2:3b`.
   - Multi-turn conversational memory buffer ($k \ge 6$) with pronoun resolution.
   - Hybrid routing: answers both exact calculation requests and qualitative semantic questions with citations.

---

## 8. How to Run the Project

1. Activate virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. Start the interactive assistant:
   ```powershell
   python main.py
   ```
3. Try sample questions:
   - *"Who are the top 5 scorers in the batch?"*
   - *"Who topped semester 4?"*
   - *"What is the overall batch average CGPA and pass percentage?"*
   - *"Show scorecard for student 1RV21CS028"*
   - *"Compare student 1RV21CS001 and 1RV21CS002"*
   - *"What is the grading policy for Grade O and Grade A+?"*
   - *"What is the hostel fee?"* (Tests refusal on out-of-context query)
