# -*- coding: utf-8 -*-
"""
Hybrid retrieval (BM25 + FAISS) + LLM re-ranking for your RAG index.
Drops in as a lightweight dependency for chatgpt_diagnosis_api_v2_mt.py
"""

import os, json, time

# Fix for old sqlite3 (must be before any chromadb import)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from openai import OpenAI
try:
    from openai import AsyncOpenAI
    HAS_ASYNC_OPENAI = True
except ImportError:
    HAS_ASYNC_OPENAI = False

# ==== FAISS & embeddings ====
# 这两行你现在的环境有就不用动
from langchain_openai import OpenAIEmbeddings
try:
    # 新版路径（0.2+ 拆到 community）
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:
    # 老版路径
    from langchain.vectorstores import FAISS

# ==== retrievers ====
# BM25 先用 community（新），没有就用老的
try:
    from langchain_community.retrievers import BM25Retriever
except ModuleNotFoundError:
    from langchain.retrievers import BM25Retriever

# EnsembleRetriever 兼容性处理
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        # 如果都没有，定义一个占位符，避免 import 错误
        EnsembleRetriever = None

# ==== Document ====
try:
    from langchain.schema import Document
except Exception:
    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.docstore.document import Document

import pickle


@dataclass
class HybridLLMRerankRetriever:
    index_dir: str = "./rag_index"
    base_url: str = "https://yunwu.ai/v1"
    api_key: Optional[str] = None
    llm_model: str = "gpt-4o"
    embed_model: str = "text-embedding-3-large"

    faiss_k: int = 15
    bm25_k: int = 15
    ensemble_weights: Tuple[float, float] = (0.5, 0.5)

    rerank_k: int = 10
    batch_size: int = 16
    max_retries: int = 4
    retry_base: float = 1.6
    use_async: bool = True  # 是否使用异步重排（如果支持）
    batch_embedding: bool = True  # 是否使用批量向量化
    embed_workers: int = 10  # 向量化并发线程数
    rerank_workers: int = 10  # 重排并发线程数

    def __post_init__(self):
        if not self.api_key:
            raise RuntimeError("Please set YUNWU_API_KEY or pass api_key.")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        # 异步客户端（如果支持）
        if HAS_ASYNC_OPENAI and self.use_async:
            self.async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.async_client = None
        self.embeddings = OpenAIEmbeddings(
            model=self.embed_model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.vs = FAISS.load_local(
            folder_path=self.index_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
            normalize_L2=True,  # 显式保证查询时向量单位化，内积≈余弦
        )

        pkl = os.path.join(self.index_dir, "docs.pkl")
        if not os.path.exists(pkl):
            raise FileNotFoundError(f"docs.pkl not found under {self.index_dir}")
        with open(pkl, "rb") as f:
            self.docs_all: List[Document] = pickle.load(f)

        self.bm25 = BM25Retriever.from_documents(self.docs_all)
        self.bm25.k = self.bm25_k

        self.faiss = self.vs.as_retriever(search_kwargs={"k": self.faiss_k})

        # self.hybrid = EnsembleRetriever(...)  # 移除对 EnsembleRetriever 的依赖，改用手动融合
        
        # 线程锁：保护 BM25 k 值的修改（细粒度锁，只锁 k 修改和检索调用）
        self._bm25_lock = threading.Lock()
        # 线程池：用于并行执行 BM25 和 FAISS 检索
        self._search_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid_search")
        # 线程池：用于并发向量化（多个查询的向量化可以并发）
        self._embed_executor = ThreadPoolExecutor(max_workers=self.embed_workers, thread_name_prefix="embedding")
        # 线程池：用于并发重排（多个查询的重排可以并发）
        self._rerank_executor = ThreadPoolExecutor(max_workers=self.rerank_workers, thread_name_prefix="rerank")

    # =========================
    # Minimal-change: add initial_k + two-stage path (backward compatible)
    # =========================
    def search(self, query: str, top_k: int = 8, initial_k: Optional[int] = None) -> List[Document]:
        """
        Two-stage: large recall initial_k -> LLM re-rank -> take top_k.
        Backward compatible: if initial_k is None, behaves like old version
        (initial_k = max(top_k, self.rerank_k)).
        """
        initial_k = initial_k or max(top_k, self.rerank_k)
        docs = self._two_stage(query, initial_k=initial_k, final_k=top_k, return_scores=False)
        return docs

    def search_with_scores(self, query: str, top_k: int = 8, initial_k: Optional[int] = None) -> List[Tuple[float, Document]]:
        """
        Same as search(), but returns (score, doc) with LLM re-rank scores (0–100).
        Backward compatible when initial_k is None.
        """
        initial_k = initial_k or max(top_k, self.rerank_k)
        scored = self._two_stage(query, initial_k=initial_k, final_k=top_k, return_scores=True)
        return scored

    # ---------- internals ----------
    def _two_stage(
        self,
        query: str,
        initial_k: int,
        final_k: int,
        return_scores: bool = False,
        bm25_k: Optional[int] = None,
        faiss_k: Optional[int] = None,
    ):
        """
        Stage-1: BM25@bm25_k + FAISS@faiss_k -> ensemble -> dedup
        Stage-2: LLM re-rank -> keep top final_k
        Note: bm25_k/faiss_k default to max(config, initial_k*2) to ensure a sufficiently large fusion pool.
        """
        # 1) Adjust per-retriever K (override defaults for this call)
        bm25_k = bm25_k or max(self.bm25_k, initial_k)
        faiss_k = faiss_k or max(self.faiss_k, initial_k)
        
        # 2) 并行执行 BM25 和 FAISS 检索（手动融合版本）
        try:
            candidates = self._parallel_hybrid_search(query, bm25_k, faiss_k)
        except Exception as e:
            # 如果并行失败，尝试串行分别检索再融合
            import warnings
            warnings.warn(f"并行检索失败，回退到串行版本: {e}")
            try:
                # 串行检索
                bm25_docs = self._bm25_search(query, bm25_k)
                
                # FAISS 检索
                if hasattr(self.vs, 'similarity_search_with_score_by_vector'):
                    # 尝试向量化后搜索
                    try:
                        qvec = self._embed_query(query)
                        import numpy as np
                        qvec_np = np.array([qvec], dtype="float32")
                        norm = np.linalg.norm(qvec_np)
                        if norm > 0: qvec_np = qvec_np / norm
                        docs_with_scores = self.vs.similarity_search_with_score_by_vector(qvec_np[0], k=faiss_k)
                        faiss_docs = [doc for doc, score in docs_with_scores]
                    except Exception:
                        faiss_docs = self.vs.similarity_search(query, k=faiss_k)
                else:
                    faiss_docs = self.vs.similarity_search(query, k=faiss_k)
                
                # 手动融合
                all_docs = {}
                for doc in bm25_docs:
                    uid = doc.metadata.get("uid") or f"{doc.metadata.get('source')}:{hash(doc.page_content)}"
                    if uid not in all_docs:
                        all_docs[uid] = {"doc": doc, "bm25_score": 1.0, "faiss_score": 0.0}
                    else:
                        all_docs[uid]["bm25_score"] = 1.0
                
                for doc in faiss_docs:
                    uid = doc.metadata.get("uid") or f"{doc.metadata.get('source')}:{hash(doc.page_content)}"
                    if uid not in all_docs:
                        all_docs[uid] = {"doc": doc, "bm25_score": 0.0, "faiss_score": 1.0}
                    else:
                        all_docs[uid]["faiss_score"] = 1.0
                
                w_bm25, w_faiss = self.ensemble_weights
                scored_docs = [
                    (w_bm25 * item["bm25_score"] + w_faiss * item["faiss_score"], item["doc"])
                    for item in all_docs.values()
                ]
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                candidates = [doc for score, doc in scored_docs]
                
            except Exception as e2:
                warnings.warn(f"串行检索也失败: {e2}")
                candidates = []
        
        uniq = self._dedup_docs(candidates)

        # 3) LLM re-rank, keep initial_k
        if return_scores:
            scored = self._llm_rerank_with_scores(query, uniq, keep_top_k=initial_k)
            return scored[:final_k]
        else:
            ranked = self._llm_rerank(query, uniq, keep_top_k=initial_k)
            return ranked[:final_k]
    
    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """BM25 检索（线程安全）"""
        with self._bm25_lock:
            old_k = self.bm25.k
            try:
                self.bm25.k = k
                try:
                    docs = self.bm25.invoke(query)
                except AttributeError:
                    docs = self.bm25.get_relevant_documents(query)
            finally:
                self.bm25.k = old_k
        return docs
    
    def _embed_query(self, query: str) -> List[float]:
        """向量化查询（线程安全，可并发）"""
        return self.embeddings.embed_query(query)
    
    def _faiss_search(self, query: str, k: int, query_vector: Optional[List[float]] = None) -> List[Document]:
        """FAISS 检索（线程安全，支持预向量化）"""
        # 如果提供了预向量化的向量，尝试直接使用向量搜索
        if query_vector is not None:
            try:
                import numpy as np
                query_vec = np.array([query_vector], dtype="float32")
                # 归一化
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm
                # 尝试使用向量直接搜索（如果 FAISS 支持）
                if hasattr(self.vs, 'similarity_search_with_score_by_vector'):
                    docs_with_scores = self.vs.similarity_search_with_score_by_vector(query_vec[0], k=k)
                    docs = [doc for doc, score in docs_with_scores]
                    return docs
            except Exception:
                # 如果失败，回退到使用查询文本
                pass
        
        # 使用查询文本（会内部向量化，但向量化已经在外部并发完成）
        temp_faiss = self.vs.as_retriever(search_kwargs={"k": k})
        try:
            docs = temp_faiss.invoke(query)
        except AttributeError:
            docs = temp_faiss.get_relevant_documents(query)
        return docs
    
    def _get_or_create_async_loop(self):
        """获取或创建异步事件循环（线程安全，每个线程独立）"""
        # 使用线程本地存储，每个线程有自己的事件循环
        if not hasattr(threading.current_thread(), '_async_loop'):
            try:
                # 尝试获取当前线程的事件循环
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # 如果没有事件循环，创建新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            threading.current_thread()._async_loop = loop
        return threading.current_thread()._async_loop
    
    def _batch_embed_queries(self, queries: List[str]) -> List[List[float]]:
        """
        批量向量化多个查询（如果 API 支持批量）
        """
        if not self.batch_embedding or len(queries) == 1:
            # 单个查询或禁用批量，逐个处理
            return [self.embeddings.embed_query(q) for q in queries]
        
        try:
            # 尝试批量向量化
            vectors = self.embeddings.embed_documents(queries)
            return vectors
        except Exception as e:
            # 如果批量失败，回退到逐个处理
            import warnings
            warnings.warn(f"批量向量化失败，回退到逐个处理: {e}")
            return [self.embeddings.embed_query(q) for q in queries]
    
    def _parallel_hybrid_search(self, query: str, bm25_k: int, faiss_k: int, query_vector: Optional[List[float]] = None) -> List[Document]:
        """
        并行执行 BM25 和 FAISS 检索，然后混合融合
        这是优化版本，可以并行执行两个检索器
        如果提供了 query_vector，可以避免重复向量化
        """
        # 并行执行向量化（如果需要）、BM25 和 FAISS 检索
        if query_vector is None:
            # 需要向量化，使用线程池并发
            embed_future = self._embed_executor.submit(self._embed_query, query)
            bm25_future = self._search_executor.submit(self._bm25_search, query, bm25_k)
            # 等待向量化完成
            query_vector = embed_future.result()
            faiss_future = self._search_executor.submit(self._faiss_search, query, faiss_k, query_vector)
        else:
            # 已有向量，直接使用
            bm25_future = self._search_executor.submit(self._bm25_search, query, bm25_k)
            faiss_future = self._search_executor.submit(self._faiss_search, query, faiss_k, query_vector)
        
        # 等待两个检索完成
        bm25_docs = bm25_future.result()
        faiss_docs = faiss_future.result()
        
        # 混合融合（使用 EnsembleRetriever 的逻辑）
        # 简单实现：合并去重，然后按权重排序
        all_docs = {}
        for doc in bm25_docs:
            uid = doc.metadata.get("uid") or f"{doc.metadata.get('source')}:{hash(doc.page_content)}"
            if uid not in all_docs:
                all_docs[uid] = {"doc": doc, "bm25_score": 1.0, "faiss_score": 0.0}
            else:
                all_docs[uid]["bm25_score"] = 1.0
        
        for doc in faiss_docs:
            uid = doc.metadata.get("uid") or f"{doc.metadata.get('source')}:{hash(doc.page_content)}"
            if uid not in all_docs:
                all_docs[uid] = {"doc": doc, "bm25_score": 0.0, "faiss_score": 1.0}
            else:
                all_docs[uid]["faiss_score"] = 1.0
        
        # 按权重计算综合分数
        w_bm25, w_faiss = self.ensemble_weights
        scored_docs = [
            (
                w_bm25 * item["bm25_score"] + w_faiss * item["faiss_score"],
                item["doc"]
            )
            for item in all_docs.values()
        ]
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs]

    def _dedup_docs(self, docs: List[Document]) -> List[Document]:
        seen = set()
        out = []
        for d in docs:
            uid = d.metadata.get("uid") or f"{d.metadata.get('source')}:{hash(d.page_content)}"
            if uid in seen:
                continue
            seen.add(uid)
            out.append(d)
        return out

    def _build_rerank_prompt(self, query: str, docs: List[Document]) -> str:
        def trunc(t: str, maxlen: int = 1200):
            t = t or ""
            return t if len(t) <= maxlen else t[:maxlen] + " ..."
        items = []
        for i, d in enumerate(docs):
            items.append({
                "id": i,
                "source": d.metadata.get("source", ""),
                "snippet": trunc(d.page_content),
            })
        instr = (
            "You are a clinical evidence re-ranker. Given a patient case query and a list of text snippets, "
            "assign each snippet a relevance score in [0,100] for supporting diagnosis/management of THIS case. "
            "Return ONLY a JSON array of numbers in the SAME order as the inputs—no extra text.\n\n"
            "Scoring (0–100):\n"
            "90–100: Strong match on key symptoms/signs/labs/imaging; directly supports reasoning or decisions.\n"
            "70–89: Generally relevant; covers some key points.\n"
            "40–69: Broadly related topic but key points are missing or imprecise.\n"
            "1–39: Weakly related or mostly unrelated to the query.\n"
            "0: Empty content or clearly irrelevant.\n\n"
            "Rules:\n"
            "• Use ONLY information present in each snippet; do NOT hallucinate.\n"
            "• If partially relevant, score proportionally.\n"
            "• Output must be a JSON array of length N (N = number of snippets), numbers only."
        )
        return json.dumps(
            {"instruction": instr, "query": query, "snippets": items},
            ensure_ascii=False
        )
    
    def _call_llm_with_retries(self, payload_json: str) -> List[float]:
        """同步 LLM 调用（带重试）"""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You output only valid JSON arrays, no prose."},
                        {"role": "user", "content": payload_json},
                    ],
                    temperature=0.0,
                )
                txt = resp.choices[0].message.content.strip()
                if txt.startswith("```"):
                    txt = txt.strip("` \n")
                    if txt.lower().startswith("json"):
                        txt = txt[4:].strip()
                scores = json.loads(txt)
                # ---- minimal robustness: length align to N snippets ----
                n = len(json.loads(payload_json)["snippets"])
                if not isinstance(scores, list):
                    raise ValueError("LLM did not return a JSON array.")
                scores = [float(s) if isinstance(s, (int, float)) else 0.0 for s in scores]
                if len(scores) > n:
                    scores = scores[:n]
                elif len(scores) < n:
                    scores += [0.0] * (n - len(scores))
                return scores
            except Exception:
                if attempt >= self.max_retries:
                    n = len(json.loads(payload_json)["snippets"])
                    return [float(n - i) for i in range(n)]
                time.sleep(self.retry_base ** attempt)
        return []
    
    async def _call_llm_async_with_retries(self, payload_json: str) -> List[float]:
        """异步 LLM 调用（带重试）"""
        if not self.async_client:
            # 如果没有异步客户端，回退到同步调用
            return self._call_llm_with_retries(payload_json)
        
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self.async_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You output only valid JSON arrays, no prose."},
                        {"role": "user", "content": payload_json},
                    ],
                    temperature=0.0,
                )
                txt = resp.choices[0].message.content.strip()
                if txt.startswith("```"):
                    txt = txt.strip("` \n")
                    if txt.lower().startswith("json"):
                        txt = txt[4:].strip()
                scores = json.loads(txt)
                # ---- minimal robustness: length align to N snippets ----
                n = len(json.loads(payload_json)["snippets"])
                if not isinstance(scores, list):
                    raise ValueError("LLM did not return a JSON array.")
                scores = [float(s) if isinstance(s, (int, float)) else 0.0 for s in scores]
                if len(scores) > n:
                    scores = scores[:n]
                elif len(scores) < n:
                    scores += [0.0] * (n - len(scores))
                return scores
            except Exception as e:
                if attempt >= self.max_retries:
                    n = len(json.loads(payload_json)["snippets"])
                    return [float(n - i) for i in range(n)]
                await asyncio.sleep(self.retry_base ** attempt)
        return []
    
    async def _llm_rerank_async(self, query: str, docs: List[Document], keep_top_k: int) -> List[Document]:
        """异步 LLM 重排"""
        if not docs:
            return []
        payload = self._build_rerank_prompt(query, docs)
        scores = await self._call_llm_async_with_retries(payload)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:keep_top_k]]
    
    async def _llm_rerank_with_scores_async(self, query: str, docs: List[Document], keep_top_k: int) -> List[Tuple[float, Document]]:
        """异步 LLM 重排（带分数）"""
        if not docs:
            return []
        payload = self._build_rerank_prompt(query, docs)
        scores = await self._call_llm_async_with_retries(payload)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:keep_top_k]
    
    def _run_async(self, coro):
        """在线程中运行异步函数（用于从同步代码调用异步方法）"""
        if not self.async_client:
            return None
        
        try:
            loop = self._get_or_create_async_loop()
            if loop.is_running():
                # 如果事件循环正在运行（在异步上下文中），使用 run_coroutine_threadsafe
                # 但这需要另一个线程的事件循环，这里简化处理：创建新线程运行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: asyncio.run(coro))
                    return future.result()
            else:
                # 如果事件循环没有运行，直接运行
                return loop.run_until_complete(coro)
        except Exception as e:
            import warnings
            warnings.warn(f"异步执行失败: {e}")
            return None

    def _llm_rerank_sync(self, query: str, docs: List[Document], keep_top_k: int) -> List[Document]:
        """同步 LLM 重排（用于线程池并发）"""
        if not docs:
            return []
        payload = self._build_rerank_prompt(query, docs)
        scores = self._call_llm_with_retries(payload)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:keep_top_k]]
    
    def _llm_rerank(self, query: str, docs: List[Document], keep_top_k: int) -> List[Document]:
        """LLM 重排（自动选择同步或异步）"""
        if not docs:
            return []
        
        # 如果支持异步且启用，使用异步版本
        if self.async_client and self.use_async:
            try:
                result = self._run_async(self._llm_rerank_async(query, docs, keep_top_k))
                if result is not None:
                    return result
            except Exception as e:
                import warnings
                warnings.warn(f"异步重排失败，回退到同步版本: {e}")
        
        # 回退到同步版本
        return self._llm_rerank_sync(query, docs, keep_top_k)

    def _llm_rerank_with_scores_sync(self, query: str, docs: List[Document], keep_top_k: int) -> List[Tuple[float, Document]]:
        """同步 LLM 重排（带分数，用于线程池并发）"""
        if not docs:
            return []
        payload = self._build_rerank_prompt(query, docs)
        scores = self._call_llm_with_retries(payload)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:keep_top_k]
    
    def _llm_rerank_with_scores(self, query: str, docs: List[Document], keep_top_k: int) -> List[Tuple[float, Document]]:
        """LLM 重排（带分数，自动选择同步或异步）"""
        if not docs:
            return []
        
        # 如果支持异步且启用，使用异步版本
        if self.async_client and self.use_async:
            try:
                result = self._run_async(self._llm_rerank_with_scores_async(query, docs, keep_top_k))
                if result is not None:
                    return result
            except Exception as e:
                import warnings
                warnings.warn(f"异步重排失败，回退到同步版本: {e}")
        
        # 回退到同步版本
        return self._llm_rerank_with_scores_sync(query, docs, keep_top_k)
    
    def batch_search(self, queries: List[str], top_k: int = 8, initial_k: Optional[int] = None) -> List[List[Document]]:
        """
        批量检索多个查询（使用批量向量化、并发向量化、并发重排）
        这是性能优化版本，适合批量处理多个查询
        """
        if not queries:
            return []
        
        initial_k = initial_k or max(top_k, self.rerank_k)
        
        # 步骤1: 批量或并发向量化查询
        query_vectors = None
        if self.batch_embedding and len(queries) > 1:
            try:
                # 尝试批量向量化（最快）
                query_vectors = self._batch_embed_queries(queries)
            except Exception:
                # 批量失败，使用并发向量化
                with ThreadPoolExecutor(max_workers=min(len(queries), self.embed_workers)) as executor:
                    embed_futures = [executor.submit(self._embed_query, q) for q in queries]
                    query_vectors = [f.result() for f in embed_futures]
        else:
            # 单个查询或禁用批量，逐个向量化
            query_vectors = [self._embed_query(q) for q in queries]
        
        # 步骤2: 并发执行检索（BM25 + FAISS）
        def search_with_vector(query: str, qvec: List[float]):
            bm25_k = max(self.bm25_k, initial_k)
            faiss_k = max(self.faiss_k, initial_k)
            candidates = self._parallel_hybrid_search(query, bm25_k, faiss_k, qvec)
            return self._dedup_docs(candidates)
        
        with ThreadPoolExecutor(max_workers=min(len(queries), self.embed_workers)) as executor:
            search_futures = [executor.submit(search_with_vector, q, qvec) 
                            for q, qvec in zip(queries, query_vectors)]
            candidates_list = [f.result() for f in search_futures]
        
        # 步骤3: 并发执行重排（使用线程池或异步）
        if self.async_client and self.use_async and len(queries) > 1:
            # 使用异步批量重排（最快）
            try:
                loop = self._get_or_create_async_loop()
                if not loop.is_running():
                    async def batch_rerank_all():
                        rerank_tasks = [
                            self._llm_rerank_async(query, candidates, keep_top_k=initial_k)
                            for query, candidates in zip(queries, candidates_list)
                        ]
                        return await asyncio.gather(*rerank_tasks)
                    
                    ranked_lists = loop.run_until_complete(batch_rerank_all())
                    return [docs[:top_k] for docs in ranked_lists]
            except Exception as e:
                import warnings
                warnings.warn(f"批量异步重排失败，回退到线程池并发: {e}")
        
        # 回退到线程池并发重排
        with ThreadPoolExecutor(max_workers=min(len(queries), self.rerank_workers)) as executor:
            rerank_futures = [executor.submit(self._llm_rerank_sync, q, candidates, initial_k)
                            for q, candidates in zip(queries, candidates_list)]
            ranked_lists = [f.result() for f in rerank_futures]
        
        return [docs[:top_k] for docs in ranked_lists]
