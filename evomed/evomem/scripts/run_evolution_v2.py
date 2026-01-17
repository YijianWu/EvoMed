#!/usr/bin/env python3
"""Run ACE evolution with internal retrieval - Evolution Version 2.0.

Evolution Version 2.0: 内部检索演化模式（仿照run_v2.py）
- 第一批次（batch 1）：从零构建经验库
- 后续批次（batch 2+）：内部检索第一批次经验 → 直接反思(TAG评估) → Curator(TAG/ADD)
- ADD操作时进行重复检查
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time
import re
from threading import Lock
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    try:
        import faiss
        FAISS_AVAILABLE = True
        print("[INFO] FAISS library available for efficient similarity search")
    except ImportError:
        faiss = None
        FAISS_AVAILABLE = False
        print("[WARN] FAISS not available, using brute force similarity search")
    SEMANTIC_DEPS_AVAILABLE = True
except ImportError:
    print("[WARN] SentenceTransformers not available, falling back to string matching")
    SentenceTransformer = None
    cosine_similarity = None
    np = None
    faiss = None
    FAISS_AVAILABLE = False
    SEMANTIC_DEPS_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (  # noqa: E402
    AdapterStepResult,
    Curator,
    EnvironmentResult,
    Generator,
    OfflineAdapter,
    Playbook,
    Reflector,
    Sample,
    TaskEnvironment,
    UniversalLLMClient,
)


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

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        """
        初始化语义检索器

        Args:
            model_name: SentenceTransformer模型名称，使用轻量级模型以提高速度
        """
        self.model_name = model_name
        if SEMANTIC_DEPS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"[INFO] Initialized semantic retriever with model: {model_name}")
            except Exception as e:
                print(f"[WARN] Failed to load SentenceTransformer model: {e}")
                print("[WARN] Falling back to simple string matching")
                self.model = None
        else:
            print("[WARN] SentenceTransformers dependencies not available, using string matching")
            self.model = None

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
                        print("[INFO] Rebuilding FAISS index for updated experience base...")
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

    def check_duplicate(self, new_content: str, threshold: float = 0.8) -> tuple[bool, str]:
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

    def _check_duplicate_brute_force(self, new_content: str, threshold: float = 0.8) -> tuple[bool, str]:
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



class InternalRetriever:
    """
    ACE内部检索器：从经验库中检索相关知识
    """

    def __init__(self, playbook: Playbook):
        self.playbook = playbook

    def search_similar(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        基于语义相似度检索经验库中的相关条目

        Args:
            query: 检索查询
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        results = []
        query_lower = query.lower()

        # 遍历所有经验条目
        for bullet_id, bullet in self.playbook._bullets.items():
            content = bullet.content.lower()

            # 计算相似度（简单实现）
            similarity = self._calculate_similarity(query_lower, content)

            if similarity > 0.1:  # 相似度阈值 - 提高质量保证
                results.append(RetrievalResult(
                    bullet_id=bullet_id,
                    content=bullet.content,
                    score=similarity,
                    source="experience"
                ))

        # 按相似度排序并返回top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        return SequenceMatcher(None, text1, text2).ratio()

    def check_duplicate(self, new_content: str, threshold: float = 0.8) -> tuple[bool, str]:
        """
        检查新内容是否与现有经验重复

        Args:
            new_content: 新经验内容
            threshold: 重复阈值

        Returns:
            (是否重复, 最相似条目的ID)
        """
        if not self.content_cache:
            return False, ""

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


@dataclass
class QuestionSample(Sample):
    """Adds a stable identifier to each sample."""

    sample_id: str = ""


class FireInvestigationEnvironment(TaskEnvironment):
    """
    环境打分版本：使用金标准模式，诊断始终正确。
    由于Generator直接使用ground_truth，Top5_Hit始终为1。
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # 金标准模式下，不需要预存Top5_Hit映射
        pass

    def evaluate(self, sample, generator_output):
        # 由于Generator直接使用ground_truth作为输出，诊断始终正确
        top5_hit = 1  # 始终正确
        status = "aligned"  # 始终对齐
        feedback = (
            f"Top5_Hit={top5_hit} → {status}. "
            "Using ground truth as diagnosis result for testing ACE reflection mechanism."
        )
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics={"Top5_Hit": top5_hit},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--excel",
        default="/gpfs/flash/home/wyj/futong/output/guilin_100K_20K_诊断_20260102_161948.xlsx",
        help="Path to the Excel file that already contains model diagnoses.",
    )
    parser.add_argument(
        "--model-path",
        default="/data/models/openai/gpt-oss-20b",
        help="Model used for Reflector/Curator (or remote model name).",
    )
    parser.add_argument(
        "--backend",
        default="transformers",
        help="LLM backend to use (transformers / openai).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the markdown report. If omitted, a timestamped file under reports/ is used.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="2,3",
        help="Comma-separated CUDA device ids to expose (default: 2,3).",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of offline adaptation epochs."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default deterministic).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N rows from the Excel file (optional).",
    )
    # 新增批次相关参数
    parser.add_argument(
        "--batch-id",
        type=int,
        required=True,
        help="Batch ID to process (1 for first batch, 2 for second batch, etc.).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of samples per batch (default: 1000).",
    )
    parser.add_argument(
        "--previous-playbook",
        default=None,
        help="Path to the playbook from previous batch (required for batch > 1).",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=5,
        help="Number of top retrieval results for internal search (default: 5).",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for duplicate detection (default: 0.95).",
    )
    return parser.parse_args()


def load_questions_batch(df: pd.DataFrame, batch_id: int, batch_size: int) -> List[QuestionSample]:
    """
    从 Excel 读取病例，并按批次封装成样本。

    Evolution Version 2.0:
    - batch 1: 从零构建经验库，使用原始数据
    - batch 2+: 使用前一批次的经验库，通过内部检索获取相关经验
    """

    def safe_get(row, colname: str) -> str:
        try:
            value = getattr(row, colname, None)
            if value is not None and pd.notna(value):
                return str(value).strip()
        except:
            pass
        return ""

    # 计算批次的样本范围
    start_idx = (batch_id - 1) * batch_size
    end_idx = min(batch_id * batch_size, len(df))

    print(f"[INFO] Batch {batch_id}: processing samples {start_idx} to {end_idx-1} (total: {end_idx - start_idx})")

    # 切片数据
    batch_df = df.iloc[start_idx:end_idx]

    samples: List[QuestionSample] = []

    for idx, row in enumerate(batch_df.itertuples(), start=start_idx):
        parts = []

        # 1) 性别
        sex = safe_get(row, "性别_clean")
        if sex:
            parts.append(f"[性别] {sex}")

        # 2) 年龄
        age = safe_get(row, "年龄_clean")
        if age:
            parts.append(f"[年龄] {age}")

        # 3) 病历
        note = safe_get(row, "病历_clean")
        if note:
            parts.append(f"[病历] {note}")

        # 4) 检验
        labs = safe_get(row, "检验结果")
        if labs:
            parts.append(f"[检验结果] {labs}")

        # 5) 检查
        exams = safe_get(row, "检查结果")
        if exams:
            parts.append(f"[检查结果] {exams}")

        question_text = "\n".join(parts).strip()

        # ground truth
        ground_truth = safe_get(row, "诊断") or None

        # model diagnosis info (直接使用金标准作为诊断结果)
        # most_likely = safe_get(row, "most_likely_diagnosis")
        # rationale = safe_get(row, "diagnostic_rationale")

        # 使用金标准作为"完美"诊断结果，测试反思演化机制
        most_likely = ground_truth or "未知诊断"
        rationale = f"基于金标准诊断：{ground_truth}。这是一个理想的诊断结果，用于测试ACE反思和演化机制。"

        samples.append(
            QuestionSample(
                sample_id=f"q{len(samples)+1:02d}",
                question=question_text,
                context="结构化后的病历/性别/年龄/检查检验，请聚焦异常项。",
                ground_truth=ground_truth,
                metadata={
                    "most_likely_diagnosis": most_likely,
                    "diagnostic_rationale": rationale,
                    "batch_id": batch_id,  # 添加批次信息
                },
            )
        )

    return samples


def evaluate_bullet_helpfulness_v2(
    reflector: Reflector,
    question: str,
    retrieved_results: List[RetrievalResult],
    ground_truth: str,
    feedback: str
) -> Dict[str, str]:
    """
    Evolution V2.0: 评估检索到的bullet的有用性

    Args:
        reflector: 反思器
        question: 诊断问题
        retrieved_results: 检索结果列表
        ground_truth: 标准答案
        feedback: 环境反馈

    Returns:
        {bullet_id: "helpful"/"harmful"/"neutral"}
    """
    if not retrieved_results:
        return {}

    # 构建playbook摘要，包含所有检索到的bullet
    excerpt_lines = []
    bullet_contents = {}

    for result in retrieved_results:
        excerpt_lines.append(f"[{result.bullet_id}] {result.content}")
        bullet_contents[result.bullet_id] = result.content

    if not excerpt_lines:
        return {result.bullet_id: "neutral" for result in retrieved_results}

    playbook_excerpt = "\n".join(excerpt_lines)

    # 构造评估提示
    eval_prompt = f"""
请评估以下检索到的经验条目对当前诊断问题的帮助程度。

诊断问题：
{question}

标准答案：
{ground_truth}

环境反馈：
{feedback}

检索到的经验条目：
{playbook_excerpt}

请对每个经验条目进行评估，返回JSON格式：
{{
  "evaluations": {{
    "条目ID": "helpful/harmful/neutral"
  }},
  "reasoning": "简要解释"
}}

评估标准：
- helpful: 经验条目提供了有用的诊断信息
- harmful: 经验条目提供了误导性信息
- neutral: 经验条目不相关或无明显影响
"""

    try:
        # 使用reflector的LLM进行评估
        response = reflector.client.generate(
            messages=[{"role": "user", "content": eval_prompt}],
            max_new_tokens=1024,
            temperature=0.0,
        )

        response_text = response.get("content", "") if isinstance(response, dict) else str(response)

        # 解析JSON响应
        try:
            result = json.loads(response_text)
            evaluations = result.get("evaluations", {})

            # 确保所有bullet_id都被评估
            final_evaluations = {}
            for result_item in retrieved_results:
                final_evaluations[result_item.bullet_id] = evaluations.get(result_item.bullet_id, "neutral")

            return final_evaluations

        except json.JSONDecodeError:
            print(f"[WARN] JSON解析失败，使用默认评估: {response_text[:200]}...")
            return {result.bullet_id: "neutral" for result in retrieved_results}

    except Exception as e:
        print(f"[WARN] 评估失败，使用默认值: {e}")
        return {result.bullet_id: "neutral" for result in retrieved_results}


def perform_internal_retrieval(
    retriever: SemanticRetriever,
    question: str,
    ground_truth: str,
    top_k: int = 5
) -> List[RetrievalResult]:
    """
    执行内部语义检索：直接检索最相关的经验

    Args:
        retriever: 语义检索器 (SemanticRetriever)
        question: 诊断结果 (用于检索)
        ground_truth: 完整病例描述 (备用)
        top_k: 返回结果数量，默认5条

    Returns:
        检索结果列表
    """
    # 直接基于诊断结果检索最相关的top_k条经验
    if question:  # 优先使用诊断结果
        query = question
    else:  # 备用使用病例描述
        query = ground_truth

    if query:
        results = retriever.search_similar(query, top_k)
        return results
    else:
        return []


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_results(results: Iterable[AdapterStepResult]) -> Dict[str, float]:
    scores = [
        step.environment_result.metrics.get("Top5_Hit", 0.0)
        for step in results
    ]
    if not scores:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": mean(scores), "min": min(scores), "max": max(scores)}


def truncate(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def run_one_sample_evolution_v2(
    sample,
    global_adapter,
    environment,
    reflector,
    curator,
    batch_id: int,
    retriever: InternalRetriever,
    retrieval_top_k: int,
    duplicate_threshold: float,
    playbook_lock: threading.Lock,
    playbook_snapshot,
    reflection_ctx_snapshot
):
    """
    Evolution V2.0 单样本处理函数（优化版：并发处理时不实时更新，最后汇总）

    - batch 1: 从零构建经验库
    - batch 2+: 内部检索 → 直接反思(TAG评估) → Curator(TAG/ADD) → ADD时检查重复
    - 返回反思结果和delta，由主函数批量应用
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # 0. 使用传入的快照，避免并发问题

            # 1. 先并行做"不会改全局状态"的部分（使用快照，不再碰全局）
            generator_output = global_adapter.generator.generate(
                question=sample.question,
                context=sample.context,
                playbook=playbook_snapshot,  # 使用playbook快照避免并发问题
                reflection=reflection_ctx_snapshot,
                **(sample.metadata or {})
            )

            env_result = environment.evaluate(sample, generator_output)
            top5_hit = env_result.metrics.get("Top5_Hit", 0)

            # ===== 批次特定的处理逻辑 =====

            # 初始化变量
            retrieved_bullet_ids = None

            if batch_id == 1:
                # 第一批次：从零构建经验库（类似run_v1.py）
                excerpt_lines = []
                _seen = set()
                for bid in generator_output.bullet_ids:
                    if bid in _seen:
                        continue
                    content = playbook_snapshot.get_bullet_content(bid)
                    if content:
                        excerpt_lines.append(f"[{bid}] {content}")
                        _seen.add(bid)
                playbook_excerpt = "\n".join(excerpt_lines)

            else:
                # 第二批次及以后：内部检索 → 直接反思 → Curator(TAG/ADD)
                print(f"[EVAL] Batch {batch_id}: Processing sample {sample.sample_id}")

                # Step 1: 内部检索第一批次的经验
                # 主要基于诊断结果进行检索
                with playbook_lock:
                    # 添加调试信息
                    query_diagnosis = env_result.ground_truth or "无诊断"
                    print(f"[RETRIEVAL] Query diagnosis: '{query_diagnosis[:50]}...'")

                retrieved_results = perform_internal_retrieval(
                    retriever=retriever,
                        question=query_diagnosis,  # 基于诊断结果检索
                        ground_truth=sample.question,  # question作为辅助
                    top_k=retrieval_top_k
                )

                print(f"[RETRIEVAL] Found {len(retrieved_results)} relevant experiences")
                if len(retrieved_results) == 0:
                    print(f"[RETRIEVAL] Available experiences in cache: {len(retriever.content_cache) if hasattr(retriever, 'content_cache') else 'N/A'}")
                if retrieved_results:
                    print(f"[RETRIEVAL] Top scores: {[f'{r.score:.3f}' for r in retrieved_results[:3]]}")
                    print(f"[RETRIEVAL] Top results: {[r.bullet_id for r in retrieved_results[:3]]}")

                # Step 2: 直接传递检索结果，让反思阶段逐条评估
                # 不构建摘要，而是让Reflector直接处理条目列表
                retrieved_bullet_ids = [r.bullet_id for r in retrieved_results]

                if retrieved_results:
                    # 为反思构建条目清单：让Reflector逐条评估检索到的经验
                    excerpt_lines = []
                    excerpt_lines.append("=== 检索到的经验条目（请逐一评估）===")
                    excerpt_lines.append("")

                    for i, result in enumerate(retrieved_results, 1):
                        excerpt_lines.append(f"【条目{i} - ID:{result.bullet_id}】")
                        excerpt_lines.append(f"内容：{result.content}")
                        excerpt_lines.append("---")

                    excerpt_lines.append("")
                    excerpt_lines.append("评估要求：")
                    excerpt_lines.append("- 对每个条目进行helpful/harmful/neutral评估")
                    excerpt_lines.append("- helpful: 对当前病例诊断有帮助")
                    excerpt_lines.append("- harmful: 可能导致误诊")
                    excerpt_lines.append("- neutral: 不相关或无影响")
                    excerpt_lines.append("- 为每个有效条目生成对应的bullet_tags条目")

                    playbook_excerpt = "\n".join(excerpt_lines)
                else:
                    # 没有检索到相关经验，跳过TAG评估
                    print("[RETRIEVAL] No relevant experiences found, skipping TAG evaluation")
                    playbook_excerpt = "(no bullets referenced)"
                    retrieved_bullet_ids = []

            # 统一的反思流程（仿照run_v2.py）
            reflection = reflector.reflect(
                question=sample.question,
                generator_output=generator_output,
                playbook=playbook_snapshot,  # 使用playbook快照避免并发问题
                ground_truth=env_result.ground_truth,
                feedback=env_result.feedback,
                max_refinement_rounds=global_adapter.max_refinement_rounds,
                playbook_excerpt=playbook_excerpt,
                allowed_ids=retrieved_bullet_ids,  # 第一批次为None，第二批次为检索到的IDs
            )

            # 在加锁前就能算出来的东西，尽量放在锁外
            question_ctx = global_adapter._question_context(sample, env_result)
            progress_str = global_adapter._progress_string(1, 1, 1, 1)

            # ===== Evolution V2.0: Curator TAG/ADD 逻辑 =====
            curator_output = curator.curate(
                reflection=reflection,
                playbook=playbook_snapshot,
                question_context=question_ctx,
                progress=progress_str,
                playbook_text="",
            )

            # Curator现在只生成ADD操作，无需过滤
            add_operations = curator_output.delta.operations

            # 对ADD操作进行重复检查（第二批次及以后）
            filtered_add_operations = []
            if batch_id == 1:
                # 第一批次：直接添加所有ADD操作，无需重复检查
                filtered_add_operations = add_operations
                for add_op in add_operations:
                    new_content = getattr(add_op, "content", "")
                    if new_content:
                        print(f"[ADD] New experience added: {new_content[:100]}...")
            else:
                # 第二批次及以后：进行重复检查
                print(f"[DUPLICATE_CHECK] Checking {len(add_operations)} ADD operations with threshold {duplicate_threshold}")
                for add_op in add_operations:
                    new_content = getattr(add_op, "content", "")

                    if new_content:
                        # 检查是否与现有经验重复
                        is_duplicate, similar_bullet_id = retriever.check_duplicate(
                            new_content, duplicate_threshold
                        )

                        if is_duplicate:
                            print(f"[DUPLICATE] ADD operation filtered (similarity > {duplicate_threshold}): similar to {similar_bullet_id}")
                            # 重复内容被过滤，不产生任何操作
                        else:
                            filtered_add_operations.append(add_op)
                            print(f"[ADD] New experience added (passed duplicate check): {new_content[:100]}...")

            # Curator只执行ADD操作
            curator_output.delta.operations = filtered_add_operations

            # 在并发处理阶段不实时更新，返回delta供主函数批量处理
            result_data = {
                'result': AdapterStepResult(
                sample=sample,
                generator_output=generator_output,
                environment_result=env_result,
                reflection=reflection,
                curator_output=curator_output,
                    playbook_snapshot="",
                ),
                'reflection': reflection,
                'curator_delta': curator_output.delta,
                'retrieved_results': retrieved_results if batch_id > 1 else None,
                'add_count': len(filtered_add_operations) if batch_id > 1 else 0,
                'duplicate_filtered': len(add_operations) - len(filtered_add_operations) if batch_id > 1 else 0,
            }

            return result_data, None

        except Exception as e:
            sample_id = getattr(sample, "sample_id", "?")
            print(f"[retry] sample {sample_id} attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(1.0)


def build_evolution_v2_report(
    args: argparse.Namespace,
    results: List[AdapterStepResult],
    playbook_text: str,
    batch_id: int,
    batch_size: int,
) -> str:
    stats = summarize_results(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines.append("# ACE Evolution V2.0 Test Report")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Batch ID: {batch_id}")
    lines.append(f"- Batch Size: {batch_size}")
    lines.append(f"- Model: `{args.model_path}`")
    lines.append(f"- CUDA devices: `{args.cuda_visible_devices}`")
    lines.append(f"- Epochs: {args.epochs}")
    lines.append(f"- Samples: {len(results)}")
    lines.append(
        f"- Top5匹配率 (avg/min/max): {stats['avg']:.2%} / "
        f"{stats['min']:.2%} / {stats['max']:.2%} (金标准模式，始终100%)"
    )

    if batch_id == 1:
        lines.append("- Mode: 从零构建经验库")
    else:
        lines.append("- Mode: 内部检索 + 评估helpful/harmful + Curator(TAG/ADD)")
        lines.append(f"- Retrieval Top-K: {args.retrieval_top_k}")
        lines.append(f"- Duplicate Threshold: {args.duplicate_threshold}")

        # 统计Evolution V2.0特有的指标
        total_add = sum(getattr(r, 'add_count', 0) for r in results)
        total_duplicate_filtered = sum(getattr(r, 'duplicate_filtered', 0) for r in results)

        lines.append(f"- ADD Operations: {total_add}")
        lines.append(f"- Duplicates Filtered: {total_duplicate_filtered}")
        lines.append("- TAG Operations: Handled by Reflector (quality assessment)")

    lines.append("")
    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| # | Top5_Hit | Question | Final Answer (truncated) | Retrieval | ADD |")
    lines.append("|---|------------|----------|--------------------------|-----------|-----|")
    for step in results:
        score = step.environment_result.metrics.get("Top5_Hit", 0.0)
        question = truncate(step.sample.question)
        final_answer = truncate(step.generator_output.final_answer or "")

        if batch_id == 1:
            retrieval_info = "N/A"
            add_count = "N/A"
        else:
            retrieved_count = len(getattr(step, 'retrieved_results', []))
            add_count = getattr(step, 'add_count', 0)
            retrieval_info = f"{retrieved_count}"

        lines.append(
            f"| {step.sample.sample_id} | {score:.2%} | {question} | {final_answer} | {retrieval_info} | {add_count} |"
        )
    lines.append("")
    lines.append("## Detailed Findings")
    lines.append("")
    for step in results:
        score = step.environment_result.metrics.get("Top5_Hit", 0.0)
        lines.append(f"### {step.sample.sample_id} — Similarity {score:.2%}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(step.sample.question)
        lines.append("")
        lines.append("**Model Final Answer**")
        lines.append("")
        lines.append(step.generator_output.final_answer or "(empty)")
        lines.append("")

        if batch_id > 1:
            retrieved_results = getattr(step, 'retrieved_results', [])
            if retrieved_results:
                lines.append("**Retrieved Results**")
                lines.append("")
                for result in retrieved_results[:3]:  # 只显示前3个
                    lines.append(f"- [{result.bullet_id}] Score: {result.score:.3f}")
                    lines.append(f"  {truncate(result.content, 100)}")
                lines.append("")

            # 在反思阶段进行TAG评估，不再有预评估

            add_count = getattr(step, 'add_count', 0)
            duplicate_filtered = getattr(step, 'duplicate_filtered', 0)

            lines.append("**Evolution Operations**")
            lines.append("")
            lines.append("- TAG operations: Handled by Reflector")
            lines.append(f"- ADD operations: {add_count}")
            lines.append(f"- Duplicates filtered: {duplicate_filtered}")
            lines.append("")

        lines.append("**Reference Answer**")
        lines.append("")
        lines.append(step.environment_result.ground_truth or "(none)")
        lines.append("")
        lines.append("**Environment Feedback**")
        lines.append("")
        lines.append(step.environment_result.feedback)
        lines.append("")
        lines.append("**Reflection Snapshot**")
        lines.append("")
        lines.append(json.dumps(step.reflection.raw, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("**Curator Operations**")
        lines.append("")
        lines.append(json.dumps(step.curator_output.raw, ensure_ascii=False, indent=2))
        lines.append("")

    # 使用字符串形式的主 playbook
    lines.append("## Final Playbook")
    lines.append("")
    lines.append(playbook_text or "(playbook is empty)")
    lines.append("")

    return "\n".join(lines)


def load_playbook_from_file(path: str) -> Playbook:
    """
    从playbook文件加载Playbook对象
    支持两种格式：
    1. 完整的报告文件（包含"## Final Playbook"部分）
    2. 单独的playbook文件（直接就是playbook内容）
    """
    pb = Playbook()
    current_section = None
    in_playbook = False
    is_direct_playbook = False  # 是否是直接的playbook文件

    if not os.path.exists(path):
        print(f"[WARN] file not found: {path}")
        return pb

    # 检查文件名来判断文件类型
    filename = os.path.basename(path)
    if filename.startswith("playbook_batch") and not filename.startswith("evolution_v2_report"):
        is_direct_playbook = True
        print(f"[INFO] Loading direct playbook file: {filename}")
    else:
        print(f"[INFO] Loading playbook from report file: {filename}")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            # 如果是直接playbook文件，直接开始解析
            if is_direct_playbook:
                in_playbook = True
            else:
                # 报告文件格式：找到 Final Playbook 开始
                if line.strip() == "## Final Playbook":
                    in_playbook = True
                    continue

            if not in_playbook:
                continue

            # 新 Section
            if line.startswith("## "):
                current_section = line[3:].strip()
                continue

            # 跳过空行
            if not line.strip():
                continue

            # Bullet 解析： - [ID] content (helpful=H, harmful=R[, neutral=N])
            m = re.match(r"- \[([^\]]+)\]\s+(.*)", line)
            if m:
                bullet_id = m.group(1).strip()
                content_full = m.group(2).strip()
                meta: Dict[str, int] = {}

                # 提取尾部计数
                m_meta = re.search(
                    r"\(helpful=(\d+),\s*harmful=(\d+)(?:,\s*neutral=(\d+))?\)\s*$",
                    content_full,
                    flags=re.IGNORECASE,
                )
                if m_meta:
                    meta["helpful"] = int(m_meta.group(1))
                    meta["harmful"] = int(m_meta.group(2))
                    if m_meta.group(3):
                        meta["neutral"] = int(m_meta.group(3))
                    # 去掉尾部计数得到纯正文
                    content = re.sub(
                        r"\(helpful=.*\)\s*$",
                        "",
                        content_full,
                        flags=re.IGNORECASE,
                    ).strip()
                else:
                    # 没有计数就直接清理常见样式
                    content = content_full.split("(helpful")[0].strip()

                pb.add_bullet(
                    section=current_section,
                    content=content,
                    bullet_id=bullet_id,
                    metadata=meta,
                )

    try:
        max_numeric = 0
        for b in pb.bullets():
            m_num = re.search(r"-(\d+)$", b.id)
            if m_num:
                val = int(m_num.group(1))
                if val > max_numeric:
                    max_numeric = val
        pb._next_id = max(pb._next_id, max_numeric)
    except Exception:
        pass

    print(f"[INFO] Loaded {len(pb.bullets())} bullets from {path}")
    return pb


# 保持向后兼容
def load_playbook_from_report(path: str) -> Playbook:
    """向后兼容函数"""
    return load_playbook_from_file(path)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 验证参数
    if args.batch_id > 1 and not args.previous_playbook:
        # 自动查找最新的第一批次playbook文件（优先使用单独保存的playbook）
        reports_dir = ROOT / "reports"
        if reports_dir.exists():
            # 优先查找单独保存的playbook文件
            batch1_playbooks = list(reports_dir.glob("playbook_batch1_*.md"))
            if batch1_playbooks:
                latest_playbook = max(batch1_playbooks, key=lambda p: p.stat().st_mtime)
                args.previous_playbook = str(latest_playbook)
                print(f"[INFO] Auto-selected playbook file: {args.previous_playbook}")
            else:
                # 如果没有单独的playbook文件，查找完整的报告文件
                batch1_reports = list(reports_dir.glob("evolution_v2_report_batch1_*.md"))
                if batch1_reports:
                    latest_report = max(batch1_reports, key=lambda p: p.stat().st_mtime)
                    args.previous_playbook = str(latest_report)
                    print(f"[INFO] Auto-selected report file (fallback): {args.previous_playbook}")
                else:
                    print("[ERROR] No batch 1 playbook or report files found in reports/ directory")
                    sys.exit(1)
        else:
            print("[ERROR] Reports directory not found and --previous-playbook not specified")
        sys.exit(1)

    excel_path = Path(args.excel)
    df = pd.read_excel(excel_path)

    # 如果指定了limit，则只使用前limit行
    if args.limit is not None and args.limit > 0:
        df = df.iloc[: args.limit]
        print(f"Using only the first {args.limit} rows from {excel_path}.")

    # 加载批次数据
    samples = load_questions_batch(df, args.batch_id, args.batch_size)
    if samples:
        print(f"[DEBUG] First sample metadata: batch_id={samples[0].metadata.get('batch_id')}")

    print(f"Loaded {len(samples)} questions for batch {args.batch_id} from {excel_path}.")
    print(
        f"Loading model for Reflector/Curator from {args.model_path} "
        f"on GPUs {args.cuda_visible_devices}..."
    )

    # 初始化LLM客户端
    client = UniversalLLMClient(
        args.model_path,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    # 初始化组件
    generator = Generator(llm=None)
    reflector = Reflector(client)
    curator = Curator(client)

    # 根据批次初始化playbook和检索器
    if args.batch_id == 1:
        # 第一批次：从空的playbook开始构建
        print("[INFO] Batch 1: Starting with empty playbook")
        global_playbook = Playbook()
        retriever = None  # 第一批次不需要检索器
    else:
        # 后续批次：从前一批次的报告加载playbook，并初始化检索器
        print(f"[INFO] Batch {args.batch_id}: Loading playbook from {args.previous_playbook}")
        global_playbook = load_playbook_from_file(args.previous_playbook)
        print(f"[INFO] Loaded playbook with {len(global_playbook._bullets)} bullets from {len(global_playbook._sections)} sections")

        # 初始化语义检索器
        print("[INFO] Initializing semantic retriever...")
        retriever = SemanticRetriever()
        print(f"[INFO] Semantic retriever initialized with model available: {retriever.model is not None}")

        # 将playbook中的所有经验添加到检索器索引
        print(f"[INFO] Indexing {len(global_playbook._bullets)} experiences...")
        indexed_count = 0
        for bullet_id, bullet in global_playbook._bullets.items():
            retriever.add_experience(bullet_id, bullet.content)
            indexed_count += 1
        print(f"[INFO] Successfully indexed {indexed_count} experiences into semantic retriever")

        # 批量构建FAISS索引（一次性完成，避免并发重建）
        if FAISS_AVAILABLE and retriever.model is not None and SEMANTIC_DEPS_AVAILABLE:
            print("[INFO] Building FAISS index for efficient retrieval...")
            retriever._build_faiss_index()
            retriever.index_needs_update = False  # 标记为已构建
            print("[INFO] FAISS index built successfully")
        else:
            print("[INFO] FAISS not available, using brute force search")

        # 验证索引是否成功
        if hasattr(retriever, 'content_cache'):
            print(f"[INFO] Retriever content_cache size: {len(retriever.content_cache)}")

        print(f"[INFO] Initialized semantic retriever with {len(global_playbook._bullets)} indexed experiences")

    # 初始化全局adapter
    global_adapter = OfflineAdapter(
        playbook=global_playbook,
        generator=generator,
        reflector=reflector,
        curator=curator,
        max_refinement_rounds=3,
    )

    # 全局锁
    playbook_lock = threading.Lock()

    # 初始化环境
    environment = FireInvestigationEnvironment(df)

    # ===== 并行化处理 =====
    max_workers = 128
    results = []
    pending_deltas = []  # 收集所有需要应用的delta

    print(f"Starting evolution v2.0 adaptation for batch {args.batch_id} in parallel...")

    # 创建快照供并发使用
    with playbook_lock:
        playbook_snapshot = global_adapter.playbook
        reflection_ctx_snapshot = global_adapter._reflection_context()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_one_sample_evolution_v2, s, global_adapter, environment, reflector, curator, args.batch_id, retriever, args.retrieval_top_k, args.duplicate_threshold, playbook_lock, playbook_snapshot, reflection_ctx_snapshot) for s in samples]

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {args.batch_id} Adapting"):
            try:
                step_data, _ = fut.result()
                results.append(step_data['result'])

                # 收集需要批量应用的delta
                if step_data['reflection']:
                    pending_deltas.append({
                        'reflection': step_data['reflection'],
                        'curator_delta': step_data['curator_delta'],
                        'retrieved_results': step_data['retrieved_results'],
                        'add_count': step_data['add_count'],
                        'duplicate_filtered': step_data['duplicate_filtered']
                    })

                # 添加Evolution V2.0特有的信息到结果
                if args.batch_id > 1:
                    step_data['result'].retrieved_results = step_data['retrieved_results']
                    step_data['result'].add_count = step_data['add_count']
                    step_data['result'].duplicate_filtered = step_data['duplicate_filtered']

            except Exception as e:
                print(f"[WARN] Sample failed: {repr(e)}")
                continue

    # ===== 批量应用所有delta =====
    print(f"[INFO] Applying {len(pending_deltas)} collected deltas...")
    total_add_operations = 0
    with playbook_lock:
        for i, delta_data in enumerate(pending_deltas):
            reflection = delta_data['reflection']
            curator_delta = delta_data['curator_delta']

            # 统计ADD操作数量
            add_count = sum(1 for op in curator_delta.operations if str(getattr(op, "type", "")).upper() == "ADD")
            total_add_operations += add_count

            # 应用TAG操作（检查reflection中是否有bullet_tags）
            if hasattr(reflection, 'bullet_tags') and reflection.bullet_tags:
                global_adapter._apply_bullet_tags(reflection)
                print(f"[TAG] Sample {i+1}: Applied {len(reflection.bullet_tags)} bullet tags")
            elif delta_data['retrieved_results']:
                print(f"[TAG] Sample {i+1}: No bullet_tags found despite {len(delta_data['retrieved_results'])} retrieved results")
            else:
                print(f"[TAG] Sample {i+1}: Skipped TAG application (no retrieved results)")

            # 应用反思更新和curator delta
            global_adapter._update_recent_reflections(reflection)

            # 详细记录delta应用过程
            if curator_delta.operations:
                print(f"[DELTA] Sample {i+1}: Applying {len(curator_delta.operations)} operations")
                for j, op in enumerate(curator_delta.operations):
                    op_type = str(getattr(op, "type", "")).upper()
                    if op_type == "ADD":
                        section = getattr(op, "section", "")
                        content = getattr(op, "content", "")[:50]
                        print(f"[ADD] Sample {i+1}-{j+1}: {section} - {content}...")
                    elif op_type == "TAG":
                        bullet_id = getattr(op, "bullet_id", "")
                        print(f"[TAG] Sample {i+1}-{j+1}: Updating {bullet_id}")
                    elif op_type == "UPDATE":
                        bullet_id = getattr(op, "bullet_id", "")
                        print(f"[UPDATE] Sample {i+1}-{j+1}: Updating {bullet_id}")

            global_playbook.apply_delta(curator_delta)

    print(f"[INFO] Total ADD operations applied: {total_add_operations}")
    print(f"[INFO] Final playbook size: {len(global_playbook._bullets)} bullets")
    print(f"[INFO] Final playbook sections: {len(global_playbook._sections)} sections")

    # 处理完成后，global_playbook就是最终版本
    combined_playbook_text = global_playbook.as_prompt() or "(playbook is empty)"

    # 生成报告
    report_markdown = build_evolution_v2_report(
        args,
        results,
        combined_playbook_text,
        args.batch_id,
        args.batch_size,
    )

    # 保存报告
    timestamp_local = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = Path(args.output)
        if output_path.exists():
            output_path = output_path.with_name(
                f"{output_path.stem}_batch{args.batch_id}_{timestamp_local}{output_path.suffix}"
            )
    else:
        output_path = ROOT / "reports" / f"evolution_v2_report_batch{args.batch_id}_{timestamp_local}.md"

    ensure_parent(output_path)
    output_path.write_text(report_markdown, encoding="utf-8")
    print(f"Report written to {output_path}")

    # 保存playbook供下一批次使用
    playbook_output_path = ROOT / "reports" / f"playbook_batch{args.batch_id}_{timestamp_local}.md"
    playbook_output_path.write_text(combined_playbook_text, encoding="utf-8")
    print(f"Playbook saved to {playbook_output_path}")


if __name__ == "__main__":
    main()
