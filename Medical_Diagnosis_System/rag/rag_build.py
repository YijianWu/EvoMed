# rag_build.py
# -*- coding: utf-8 -*-
import os, glob, pickle, hashlib, time, logging, warnings
import argparse
from typing import List, Tuple, Dict


# ====== 静默 pypdf 的 /MediaBox 等告警 ======
try:
    from pypdf.errors import PdfReadWarning
    warnings.filterwarnings("ignore", category=PdfReadWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message="Multiple definitions in dictionary.*")

# ====== 依赖 ======
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)
try:
    # 有些问题 PDF 用这个更稳
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

# ====== 日志 ======
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)

# ====== API 配置 ======
BASE_URL = os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1")
API_KEY  = os.getenv("YUNWU_API_KEY") or "sk-CCoYJEJcm2mL4YH7uRRw9DPgXQj2f8873F1D98uXtuwclUwW"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-large")

if not API_KEY:
    raise RuntimeError("环境变量 YUNWU_API_KEY 未设置。")

oai = OpenAI(base_url=BASE_URL, api_key=API_KEY)

lc_embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
)

# ====== 路径 / 参数 ======
KNOWLEDGE_ROOT = os.getenv("RAG_KB_DIR", "./腹痛指南")  # 修改默认值为实际数据目录
INDEX_DIR      = os.getenv("RAG_INDEX_DIR", "./rag_index")

CHUNK_SIZE     = int(os.getenv("RAG_CHUNK_SIZE", "256"))
CHUNK_OVERLAP  = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

# 优化参数：适度并发，兼顾速度与限流
MAX_WORKERS    = int(os.getenv("RAG_THREADS", "4"))        # 启用4线程并发
BATCH_SIZE     = int(os.getenv("RAG_BATCH", "16"))         # 每批16个文本
MAX_RETRIES    = int(os.getenv("RAG_EMBED_RETRIES", "10")) # 保持重试次数
RETRY_BASE_S   = float(os.getenv("RAG_RETRY_BASE", "2.0")) # 保持重试等待
REQUEST_DELAY  = float(os.getenv("RAG_REQUEST_DELAY", "0.1"))  # 减少请求间隔到0.1秒

# ====== 工具函数 ======
def _hash(s: str, n: int = 10) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()[:n]

def load_all_documents(root_dir: str) -> List[Document]:
    """
    递归读取 pdf/docx/txt/md。优先 PyPDFLoader，失败时（或问题 PDF）尝试 PyMuPDFLoader。
    """
    docs: List[Document] = []
    patterns = ["**/*.pdf", "**/*.PDF", "**/*.docx", "**/*.DOCX", "**/*.txt", "**/*.TXT", "**/*.md", "**/*.MD"]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))

    paths = sorted(set(paths))
    logging.info(f"扫描到 {len(paths)} 个文件")

    # ★ 在逐文件处理处加进度条
    for path in tqdm(paths, desc="Loading files"):
        try:
            if path.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(path)
                    pd_docs = loader.load()
                except Exception as e_pdf:
                    if HAS_PYMUPDF:
                        logging.warning(f"[PDF告警] PyPDF 失败，尝试 PyMuPDF：{path} -> {e_pdf}")
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
            logging.warning(f"[跳过] 读取失败：{path} -> {e}")
    return docs


def stable_uid(meta: dict, content: str, chunk_index: int) -> str:
    """
    生成真正全局唯一的块ID：
    hash(source) : page : chunk_index : hash(content)
    """
    src  = meta.get("source", "unknown")
    page = meta.get("page", meta.get("page_number", -1))  # PyPDFLoader 通常有 "page"
    return f"{_hash(src)}:{page}:{chunk_index}:{_hash(content, 8)}"

def split_by_tokens(docs: List[Document]) -> List[Document]:
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    out: List[Document] = []
    # ★ 在逐文档切分处加进度条（显示文档处理进度，不是块数）
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
    基于 uid 的去重，防止问题 PDF 产生重复页/重复块。
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
    """带重试和延迟的embedding请求"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 请求前添加延迟，避免限流
            time.sleep(REQUEST_DELAY)
            resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
            data_sorted = sorted(resp.data, key=lambda x: x.index)
            return [d.embedding for d in data_sorted]
        except Exception as e:
            # 429限流时使用更长的等待时间
            if "429" in str(e) or "Too Many" in str(e):
                wait = RETRY_BASE_S ** attempt * 2  # 限流时等更久
            else:
                wait = RETRY_BASE_S ** attempt
            if attempt >= MAX_RETRIES:
                raise
            logging.warning(f"[embed retry {attempt}/{MAX_RETRIES}] {e}，{wait:.1f}s 后重试")
            time.sleep(wait)

def _normalize_ip(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def embed_texts_in_threads(texts: List[str]) -> np.ndarray:
    """
    生成文本嵌入向量
    - MAX_WORKERS=1 时使用顺序执行（避免限流）
    - MAX_WORKERS>1 时使用多线程
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype="float32")
    
    batches: List[Tuple[int, int]] = [(i, min(i + BATCH_SIZE, n)) for i in range(0, n, BATCH_SIZE)]
    results: Dict[int, np.ndarray] = {}
    
    if MAX_WORKERS <= 1:
        # 顺序执行模式（更安全，避免限流）
        logging.info(f"使用顺序模式嵌入 {len(batches)} 个批次...")
        for s, e in tqdm(batches, desc="Embedding (sequential)"):
            vecs = np.array(_embed_batch(texts[s:e]), dtype="float32")
            results[s] = vecs
    else:
        # 多线程模式
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
        raise ValueError("嵌入数量与文档块数量不一致。")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    docstore = InMemoryDocstore({})
    index_to_docstore_id: Dict[int, str] = {}
    for i, d in enumerate(docs):
        doc_id = d.metadata["uid"]  # 使用唯一 uid
        # 理论上 dedup 后不会重复；稳妥起见还是防一手
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
    logging.info("1) 读取知识库文档中 ...")
    raw_docs = load_all_documents(KNOWLEDGE_ROOT)
    logging.info(f"   - 原始文档数：{len(raw_docs)}")

    logging.info("2) Token 切分为 %d / overlap %d ...", CHUNK_SIZE, CHUNK_OVERLAP)
    chunked_docs = split_by_tokens(raw_docs)
    logging.info(f"   - 切分后文档块数：{len(chunked_docs)}")

    # 先去重，避免问题 PDF 造成重复
    chunked_docs = dedup_docs(chunked_docs)
    logging.info(f"   - 去重后文档块数：{len(chunked_docs)}")

    texts = [d.page_content for d in chunked_docs]

    logging.info("3) 并行生成嵌入（%d 线程，batch=%d） ...", MAX_WORKERS, BATCH_SIZE)
    vecs = embed_texts_in_threads(texts)
    logging.info("   - 嵌入完成。shape=%s", tuple(vecs.shape))

    logging.info("4) 构建 FAISS 向量索引 ...")
    vs = build_faiss_store(vecs, chunked_docs)
    vs.save_local(INDEX_DIR)
    logging.info(f"   - 向量索引写入：{INDEX_DIR}")

    logging.info("5) 保存 BM25 所需文档快照 ...")
    save_docs_snapshot(chunked_docs, INDEX_DIR)
    logging.info("   - 文档快照：docs.pkl")

    logging.info("✅ 全部完成。")

def parse_args():
    p = argparse.ArgumentParser(description="Build RAG FAISS index")
    p.add_argument("--kb", default=KNOWLEDGE_ROOT, help="知识库根目录")
    p.add_argument("--out", default=INDEX_DIR, help="索引输出目录")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="切块 token 大小")
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="切块重叠 token 数")
    p.add_argument("--threads", type=int, default=MAX_WORKERS, help="并发线程数")
    p.add_argument("--batch", type=int, default=BATCH_SIZE, help="嵌入 batch 大小")
    return p.parse_args()

def main():
    global KNOWLEDGE_ROOT, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS, BATCH_SIZE
    
    args = parse_args()

    # 用命令行参数覆盖全局配置
    KNOWLEDGE_ROOT = args.kb
    INDEX_DIR = args.out
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    MAX_WORKERS = args.threads
    BATCH_SIZE = args.batch

    t0 = time.time()
    build_and_save_index()
    logging.info("总用时：%.1fs", time.time() - t0)


if __name__ == "__main__":
    main()

