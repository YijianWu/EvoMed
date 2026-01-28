# rag_build.py
# -*- coding: utf-8 -*-
import os, glob, pickle, hashlib, time, logging, warnings
import argparse
from typing import List, Tuple, Dict


# ====== Silence pypdf /MediaBox warnings ======
try:
    from pypdf.errors import PdfReadWarning
    warnings.filterwarnings("ignore", category=PdfReadWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message="Multiple definitions in dictionary.*")

# ====== Dependencies ======
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)
try:
    # Some problematic PDFs are handled more reliably by this
    from langchain_community.document_loaders import PyMuPDFLoader
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document

# Embeddings & Vectorstore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import faiss

# ====== Logging ======
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)

# ====== API Configuration ======
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY  = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

if not API_KEY:
    raise RuntimeError("Environment variable YUNWU_API_KEY is not set.")

oai = OpenAI(base_url=BASE_URL, api_key=API_KEY)

lc_embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
)

# ====== Paths / Parameters ======
KNOWLEDGE_ROOT = os.getenv("RAG_KB_DIR", "./AbdominalPainGuideline")  # Updated to reflect translated directory
INDEX_DIR      = os.getenv("RAG_INDEX_DIR", "./rag_index")

CHUNK_SIZE     = int(os.getenv("RAG_CHUNK_SIZE", "256"))
CHUNK_OVERLAP  = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

# Optimization parameters: moderate concurrency to balance speed and rate limiting
MAX_WORKERS    = int(os.getenv("RAG_THREADS", "4"))        # 4 threads for concurrency
BATCH_SIZE     = int(os.getenv("RAG_BATCH", "16"))         # 16 texts per batch
MAX_RETRIES    = int(os.getenv("RAG_EMBED_RETRIES", "10")) # Number of retries
RETRY_BASE_S   = float(os.getenv("RAG_RETRY_BASE", "2.0")) # Base wait time for retry
REQUEST_DELAY  = float(os.getenv("RAG_REQUEST_DELAY", "0.1"))  # Request interval 0.1s

# ====== Utility Functions ======
def _hash(s: str, n: int = 10) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()[:n]

def load_all_documents(root_dir: str) -> List[Document]:
    """
    Recursively read pdf/docx/txt/md. Prioritize PyPDFLoader, fallback to PyMuPDFLoader on failure.
    """
    docs: List[Document] = []
    patterns = ["**/*.pdf", "**/*.PDF", "**/*.docx", "**/*.DOCX", "**/*.txt", "**/*.TXT", "**/*.md", "**/*.MD"]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))

    paths = sorted(set(paths))
    logging.info(f"Scanned {len(paths)} files")

    # Add progress bar for file processing
    for path in tqdm(paths, desc="Loading files"):
        try:
            if path.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(path)
                    pd_docs = loader.load()
                except Exception as e_pdf:
                    if HAS_PYMUPDF:
                        logging.warning(f"[PDF Warning] PyPDF failed, trying PyMuPDF: {path} -> {e_pdf}")
                        loader = PyMuPDFLoader(path)
                        pd_docs = loader.load()
                    else:
                        raise e_pdf
                for d in pd_docs:
                    d.metadata["source"] = path
                docs.extend(pd_docs)

            elif path.lower().endswith(".docx"):
                loader = Docx2txtLoader(path)
                d = loader.load()
                for x in d:
                    x.metadata["source"] = path
                docs.extend(d)

            else:
                loader = TextLoader(path, encoding="utf-8")
                d = loader.load()
                for x in d:
                    x.metadata["source"] = path
                docs.extend(d)

        except Exception as e:
            logging.warning(f"[Skip] Failed to read: {path} -> {e}")
    return docs


def stable_uid(meta: dict, content: str, chunk_index: int) -> str:
    """
    Generate globally unique chunk ID:
    hash(source) : page : chunk_index : hash(content)
    """
    src  = meta.get("source", "unknown")
    page = meta.get("page", meta.get("page_number", -1))  # PyPDFLoader typically uses "page"
    return f"{_hash(src)}:{page}:{chunk_index}:{_hash(content, 8)}"

def split_by_tokens(docs: List[Document]) -> List[Document]:
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    out: List[Document] = []
    # Add progress bar for document splitting
    for d in tqdm(docs, desc="Token splitting"):
        chunks = splitter.split_text(d.page_content or "")
        for i, ch in enumerate(chunks):
            meta = dict(d.metadata)
            meta["page"] = meta.get("page", meta.get("page_number", meta.get("source_id", -1)))
            meta["chunk_index"] = i
            meta["uid"] = stable_uid(meta, ch, i)
            out.append(Document(page_content=ch, metadata=meta))
    return out


def dedup_docs(docs: List[Document]) -> List[Document]:
    """
    De-duplication based on uid to prevent redundant pages/chunks from problematic PDFs.
    """
    seen = set()
    uniq: List[Document] = []
    for d in docs:
        uid = d.metadata.get("uid")
        if uid in seen:
            continue
        seen.add(uid)
        uniq.append(d)
    return uniq

def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Embedding request with retries and delay"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Add delay before request to avoid rate limiting
            time.sleep(REQUEST_DELAY)
            resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
            data_sorted = sorted(resp.data, key=lambda x: x.index)
            return [d.embedding for d in data_sorted]
        except Exception as e:
            # Longer wait for 429 rate limit
            if "429" in str(e) or "Too Many" in str(e):
                wait = RETRY_BASE_S ** attempt * 2
            else:
                wait = RETRY_BASE_S ** attempt
            if attempt >= MAX_RETRIES:
                raise
            logging.warning(f"[embed retry {attempt}/{MAX_RETRIES}] {e}, retrying after {wait:.1f}s")
            time.sleep(wait)

def _normalize_ip(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def embed_texts_in_threads(texts: List[str]) -> np.ndarray:
    """
    Generate text embedding vectors
    - Sequential execution when MAX_WORKERS=1 (to avoid rate limits)
    - Multi-threaded when MAX_WORKERS>1
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype="float32")
    
    batches: List[Tuple[int, int]] = [(i, min(i + BATCH_SIZE, n)) for i in range(0, n, BATCH_SIZE)]
    results: Dict[int, np.ndarray] = {}
    
    if MAX_WORKERS <= 1:
        # Sequential execution mode (safer against rate limits)
        logging.info(f"Embedding {len(batches)} batches in sequential mode...")
        for s, e in tqdm(batches, desc="Embedding (sequential)"):
            vecs = np.array(_embed_batch(texts[s:e]), dtype="float32")
            results[s] = vecs
    else:
        # Multi-threaded mode
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_embed_batch, texts[s:e]): (s, e) for (s, e) in batches}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding (threaded)"):
                s, e = futures[fut]
                vecs = np.array(fut.result(), dtype="float32")
                results[s] = vecs
    
    dim = next(iter(results.values())).shape[1]
    out = np.zeros((n, dim), dtype="float32")
    for s, e in batches:
        out[s:e, :] = results[s]
    return _normalize_ip(out)

def build_faiss_store(embeddings: np.ndarray, docs: List[Document]) -> FAISS:
    if len(embeddings) != len(docs):
        raise ValueError("Mismatch between embedding count and document chunk count.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    docstore = InMemoryDocstore({})
    index_to_docstore_id: Dict[int, str] = {}
    for i, d in enumerate(docs):
        doc_id = d.metadata["uid"]  # Use unique uid
        # De-duplication check as a precaution
        if doc_id in docstore._dict:
            doc_id = f"{doc_id}:{_hash(str(i), 6)}"
        docstore.add({doc_id: d})
        index_to_docstore_id[i] = doc_id

    return FAISS(
        embedding_function=lc_embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        normalize_L2=True,
    )

def save_docs_snapshot(docs: List[Document], out_dir: str):
    with open(os.path.join(out_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

def build_and_save_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    logging.info("1) Reading knowledge base documents...")
    raw_docs = load_all_documents(KNOWLEDGE_ROOT)
    logging.info(f"   - Original document count: {len(raw_docs)}")

    logging.info("2) Splitting into tokens: size %d / overlap %d...", CHUNK_SIZE, CHUNK_OVERLAP)
    chunked_docs = split_by_tokens(raw_docs)
    logging.info(f"   - Document chunk count: {len(chunked_docs)}")

    # De-duplicate to handle issues from certain PDFs
    chunked_docs = dedup_docs(chunked_docs)
    logging.info(f"   - De-duplicated chunk count: {len(chunked_docs)}")

    texts = [d.page_content for d in chunked_docs]

    logging.info("3) Generating embeddings in parallel (%d threads, batch=%d)...", MAX_WORKERS, BATCH_SIZE)
    vecs = embed_texts_in_threads(texts)
    logging.info("   - Embedding complete. shape=%s", tuple(vecs.shape))

    logging.info("4) Building FAISS vector index...")
    vs = build_faiss_store(vecs, chunked_docs)
    vs.save_local(INDEX_DIR)
    logging.info(f"   - Vector index saved to: {INDEX_DIR}")

    logging.info("5) Saving document snapshots for BM25...")
    save_docs_snapshot(chunked_docs, INDEX_DIR)
    logging.info("   - Document snapshot: docs.pkl")

    logging.info("✅ All steps completed.")

def parse_args():
    p = argparse.ArgumentParser(description="Build RAG FAISS index")
    p.add_argument("--kb", default=KNOWLEDGE_ROOT, help="Knowledge base root directory")
    p.add_argument("--out", default=INDEX_DIR, help="Index output directory")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk token size")
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap token count")
    p.add_argument("--threads", type=int, default=MAX_WORKERS, help="Concurrent thread count")
    p.add_argument("--batch", type=int, default=BATCH_SIZE, help="Embedding batch size")
    return p.parse_args()

def main():
    global KNOWLEDGE_ROOT, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS, BATCH_SIZE
    
    args = parse_args()

    # Override global config with command line arguments
    KNOWLEDGE_ROOT = args.kb
    INDEX_DIR = args.out
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    MAX_WORKERS = args.threads
    BATCH_SIZE = args.batch

    t0 = time.time()
    build_and_save_index()
    logging.info("Total duration: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
