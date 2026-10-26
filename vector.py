from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os
import re

# 1. Initialize local embedding model via Ollama
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 2. Vector DB Storage Path for College Results
db_location = "./chroma_college_db"
pdf_path = "college_exam_results.pdf"

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="college_exam_results",
    persist_directory=db_location,
    embedding_function=embeddings,
)

def load_and_index_pdf(pdf_file=pdf_path):
    """
    Ingests, parses, chunks, and embeds the PDF exam results into Chroma.
    Day 5 Core Concepts:
    - Document Ingestion from PDF (PyPDF)
    - Document parsing and text chunking (RecursiveCharacterTextSplitter)
    - Source metadata tagging for exact citations (file name & page numbers)
    """
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"PDF file '{pdf_file}' not found. Run generate_college_data.py first.")

    print(f"Parsing and chunking PDF: {pdf_file}...")
    reader = PdfReader(pdf_file)
    raw_documents = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            page_num = page_idx + 1
            # Extract any USN mentioned on the page for richer metadata
            usn_matches = re.findall(r"1RV21CS\d{3}", text)
            metadata = {
                "source": os.path.basename(pdf_file),
                "page": page_num,
                "usns": ", ".join(set(usn_matches)) if usn_matches else "overview"
            }
            raw_documents.append(Document(page_content=text, metadata=metadata))

    # Semantic Chunking: preserves entire student 8-semester transcripts intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2600,
        chunk_overlap=150,
        separators=["USN: ", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Extracted {len(raw_documents)} pages into {len(chunks)} clean semantic chunks.", flush=True)

    # Add chunk IDs for tracking
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Batch indexing into Chroma
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]
        vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
        print(f"Indexed chunks {i+1} to {min(i+batch_size, len(chunks))} of {len(chunks)}...", flush=True)

    print("PDF Ingestion into Chroma Vector Store Complete!", flush=True)

# Index if database does not exist or has incomplete collection
if not os.path.exists(db_location) or vector_store._collection.count() < 100:
    # Reset any partial collection if needed
    if vector_store._collection.count() > 0:
        print("Resetting partial Chroma collection to ensure full indexing...", flush=True)
        try:
            vector_store.delete_collection()
            vector_store = Chroma(
                collection_name="college_exam_results",
                persist_directory=db_location,
                embedding_function=embeddings,
            )
        except Exception:
            pass
    load_and_index_pdf(pdf_path)
else:
    print(f"Loaded existing Chroma index ({vector_store._collection.count()} chunks stored).", flush=True)

# 3. Create Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

def retrieve_with_sources(query: str, k: int = 4):
    """
    Retrieves top-k relevant chunks and formats them with source citations.
    Fulfills Day 5 objective: Source citations and handling questions without context.
    """
    docs = vector_store.similarity_search(query, k=k)
    citations = []
    formatted_contexts = []
    
    for doc in docs:
        source = doc.metadata.get("source", "college_exam_results.pdf")
        page = doc.metadata.get("page", "?")
        citation = f"[Source: {source}, Page {page}]"
        citations.append(citation)
        formatted_contexts.append(f"--- Context from {citation} ---\n{doc.page_content}")
        
    unique_citations = list(dict.fromkeys(citations))
    combined_context = "\n\n".join(formatted_contexts)
    return combined_context, unique_citations
