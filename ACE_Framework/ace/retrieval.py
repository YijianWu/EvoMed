"""Retrieval logic for ACE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import numpy as np
from threading import Lock

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        import faiss
        FAISS_AVAILABLE = True
    except ImportError:
        faiss = None
        FAISS_AVAILABLE = False
    SEMANTIC_DEPS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None
    np = None
    faiss = None
    FAISS_AVAILABLE = False
    SEMANTIC_DEPS_AVAILABLE = False

@dataclass
class RetrievalResult:
    """检索结果"""
    bullet_id: str
    content: str
    score: float
    source: str = "experience"


class SemanticRetriever:
    """
    基于向量嵌入的语义检索器
    使用SentenceTransformer进行文本向量化，提供更好的语义匹配
    """

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2", lazy_load: bool = True):
        """
        初始化语义检索器

        Args:
            model_name: SentenceTransformer模型名称，使用轻量级模型以提高速度
            lazy_load: 是否延迟加载模型（默认True，在首次使用时加载）
        """
        self.model_name = model_name
        self.lazy_load = lazy_load
        self.model = None
        
        if SEMANTIC_DEPS_AVAILABLE and not lazy_load:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"[INFO] Initialized semantic retriever with model: {model_name}")
            except Exception as e:
                print(f"[WARN] Failed to load SentenceTransformer model: {e}")
                print("[WARN] Falling back to simple string matching")

        self.embeddings_cache = {}
        self.content_cache = {}

        # FAISS索引相关
        self.faiss_index = None
        self.index_bullet_ids = []  # 保持ID与向量的对应关系
        self.embedding_dim = None  # 向量维度
        self.index_needs_update = False  # 标记索引是否需要更新
        self.index_lock = Lock()  # 保护FAISS索引的线程锁

    def _build_faiss_index(self):
        """构建FAISS索引"""
        if not FAISS_AVAILABLE or not self.content_cache:
            return

        try:
            # 获取所有embeddings
            embeddings = []
            bullet_ids = []

            for bullet_id, content in self.content_cache.items():
                emb = self._get_embedding(content)
                if emb is not None:
                    embeddings.append(emb)
                    bullet_ids.append(bullet_id)

            if not embeddings:
                return

            # 转换为numpy数组
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.embedding_dim = embeddings_array.shape[1]

            # 创建FAISS索引
            if self.embedding_dim <= 768:  # 对于中等维度，使用IndexIVFFlat
                nlist = min(100, max(4, len(embeddings) // 39))  # IVF参数
                quantizer = faiss.IndexFlatIP(self.embedding_dim)  # 内积索引
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                # 训练索引
                self.faiss_index.train(embeddings_array)
            else:
                # 对于高维度，使用IndexFlatIP（暴力搜索但内存友好）
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            # 添加向量到索引
            self.faiss_index.add(embeddings_array)
            self.index_bullet_ids = bullet_ids

            print(f"[INFO] Built FAISS index with {len(bullet_ids)} vectors, dimension: {self.embedding_dim}")

        except Exception as e:
            print(f"[WARN] Failed to build FAISS index: {e}, falling back to brute force")
            self.faiss_index = None

    def _rebuild_faiss_index(self):
        """重新构建FAISS索引"""
        self.faiss_index = None
        self.index_bullet_ids = []
        self._build_faiss_index()

    def _get_embedding(self, text: str):
        """获取文本的向量嵌入（带缓存）"""
        if text not in self.embeddings_cache:
            # 延迟加载模型
            if self.model is None and self.lazy_load and SEMANTIC_DEPS_AVAILABLE:
                try:
                    print(f"[INFO] Lazy loading SentenceTransformer model: {self.model_name}")
                    self.model = SentenceTransformer(self.model_name)
                except Exception as e:
                    print(f"[WARN] Failed to lazy load model: {e}")
                    self.lazy_load = False  # 不再尝试加载
            
            if self.model is not None and SEMANTIC_DEPS_AVAILABLE:
                # 清理文本，保留关键信息
                clean_text = self._preprocess_text(text)
                self.embeddings_cache[text] = self.model.encode([clean_text])[0]
            else:
                # fallback: 使用简单的数值表示（用于字符串相似度计算）
                self.embeddings_cache[text] = hash(text) % 1000  # 简单的数值表示
        return self.embeddings_cache[text]

    def _preprocess_text(self, text: str) -> str:
        """预处理文本，提高检索质量"""
        # 保留诊断关键词，过滤噪声
        text = text.lower()

        # 提取关键模式：疾病名称、症状描述
        import re

        # 保留中文字符、英文单词、数字
        clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        # 压缩连续空格
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def add_experience(self, bullet_id: str, content: str):
        """添加经验条目到检索索引"""
        self.content_cache[bullet_id] = content
        # 预计算embedding
        self._get_embedding(content)
        # 标记FAISS索引需要更新
        self.index_needs_update = True

    def search_similar(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        基于语义相似度检索最相关的经验

        Args:
            query: 检索查询
            top_k: 返回结果数量

        Returns:
            检索结果列表，按相似度排序
        """
        if not self.content_cache:
            return []

        if self.model is None:
            # Fallback to simple string matching
            return self._fallback_search(query, top_k)

        if self.model is not None and SEMANTIC_DEPS_AVAILABLE:
            try:
                # 检查并重建FAISS索引（如果需要）- 使用线程锁保护
                with self.index_lock:
                    if self.index_needs_update or self.faiss_index is None:
                        # print("[INFO] Rebuilding FAISS index for updated experience base...")
                        self._rebuild_faiss_index()
                        self.index_needs_update = False

                # 优先使用FAISS进行高效检索
                if self.faiss_index is not None and FAISS_AVAILABLE:
                    return self._faiss_search(query, top_k)
                else:
                    # 回退到暴力搜索
                    return self._brute_force_search(query, top_k)

            except Exception as e:
                print(f"[WARN] Semantic search failed: {e}, falling back to string matching")
                return self._fallback_search(query, top_k)
        else:
            # 没有语义依赖，直接使用字符串匹配
            return self._fallback_search(query, top_k)

    def _faiss_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """使用FAISS进行高效相似度检索"""
        try:
            # 获取查询向量
            query_emb = self._get_embedding(query)
            if query_emb is None:
                return []

            query_array = np.array([query_emb], dtype=np.float32)

            # FAISS搜索 - 使用线程锁保护
            with self.index_lock:
                if self.faiss_index is None:
                    print("[WARN] FAISS index is None, falling back to brute force")
                    return self._brute_force_search(query, top_k)

                scores, indices = self.faiss_index.search(query_array, min(top_k, len(self.index_bullet_ids)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.index_bullet_ids):  # 有效索引
                    bullet_id = self.index_bullet_ids[idx]
                    content = self.content_cache.get(bullet_id, "")

                    results.append(RetrievalResult(
                        bullet_id=bullet_id,
                        content=content,
                        score=float(score),  # FAISS返回的是内积分数
                        source="experience"
                    ))

            # 按分数降序排序（FAISS可能不保证完全排序）
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        except Exception as e:
            print(f"[WARN] FAISS search failed: {e}, falling back to brute force")
            return self._brute_force_search(query, top_k)

    def _brute_force_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """暴力搜索方法（原始实现）"""
        try:
            # 获取查询的embedding
            query_emb = self._get_embedding(query)

            results = []
            for bullet_id, content in self.content_cache.items():
                content_emb = self._get_embedding(content)

                # 计算余弦相似度
                similarity = cosine_similarity([query_emb], [content_emb])[0][0]

                results.append(RetrievalResult(
                    bullet_id=bullet_id,
                    content=content,
                    score=float(similarity),
                    source="experience"
                ))

            # 按相似度降序排序
            results.sort(key=lambda x: x.score, reverse=True)

            # 返回top_k结果
            return results[:top_k]

        except Exception as e:
            print(f"[WARN] Brute force search failed: {e}, falling back to string matching")
            return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """后备方案：使用简单的字符串相似度匹配"""
        results = []
        query_lower = query.lower()

        for bullet_id, content in self.content_cache.items():
            content_lower = content.lower()
            similarity = SequenceMatcher(None, query_lower, content_lower).ratio()

            if similarity > 0.5:  # 相似度阈值 - 提高质量保证
                results.append(RetrievalResult(
                    bullet_id=bullet_id,
                    content=content,
                    score=float(similarity),
                    source="experience"
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def check_duplicate(self, new_content: str, threshold: float = 0.8) -> Tuple[bool, str]:
        """
        检查新内容是否与现有经验重复（FAISS优化版本）

        Args:
            new_content: 新经验内容
            threshold: 重复阈值

        Returns:
            (是否重复, 最相似条目的ID)
        """
        if not self.content_cache:
            return False, ""

        # 检查并重建FAISS索引（如果需要）- 使用线程锁保护
        with self.index_lock:
            if self.index_needs_update or self.faiss_index is None:
                self._rebuild_faiss_index()
                self.index_needs_update = False

        # 优先使用FAISS进行高效重复检查
        if self.faiss_index is not None and FAISS_AVAILABLE and self.model is not None and SEMANTIC_DEPS_AVAILABLE:
            try:
                new_emb = self._get_embedding(new_content)
                if new_emb is None:
                    return False, ""

                query_array = np.array([new_emb], dtype=np.float32)

                # 搜索最相似的1个结果 - 使用线程锁保护
                with self.index_lock:
                    scores, indices = self.faiss_index.search(query_array, 1)

                if indices[0][0] < len(self.index_bullet_ids):
                    best_score = float(scores[0][0])
                    best_bullet_id = self.index_bullet_ids[indices[0][0]]
                    return best_score >= threshold, best_bullet_id

            except Exception as e:
                print(f"[WARN] FAISS duplicate check failed: {e}, falling back to brute force")

        # 回退到暴力检查
        return self._check_duplicate_brute_force(new_content, threshold)

    def _check_duplicate_brute_force(self, new_content: str, threshold: float = 0.8) -> Tuple[bool, str]:
        """暴力检查重复（原始实现）"""
        new_content_clean = self._preprocess_text(new_content)
        best_score = 0.0
        best_bullet_id = ""

        for bullet_id, existing_content in self.content_cache.items():
            existing_clean = self._preprocess_text(existing_content)

            # 计算相似度
            if self.model is not None and SEMANTIC_DEPS_AVAILABLE:
                # 使用语义相似度
                new_emb = self._get_embedding(new_content)
                existing_emb = self._get_embedding(existing_content)
                similarity = cosine_similarity([new_emb], [existing_emb])[0][0]
            else:
                # 使用字符串相似度
                similarity = SequenceMatcher(None, new_content_clean, existing_clean).ratio()

            if similarity > best_score:
                best_score = similarity
                best_bullet_id = bullet_id

        return best_score >= threshold, best_bullet_id

