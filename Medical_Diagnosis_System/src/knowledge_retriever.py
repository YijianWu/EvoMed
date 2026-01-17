# -*- coding: utf-8 -*-
"""
知识检索服务

整合三大知识来源：
1. RAG医学指南检索 (BM25 + FAISS混合检索 + LLM重排)
2. 经验库检索 (A-Mem系统)
3. 病例库检索 (ACE系统)

用于为Step-2/Step-3提供参考资料
"""

import os
import sys

# Fix for old sqlite3 (must be before any chromadb import)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# API配置
API_BASE_URL = "https://yunwu.ai/v1"
API_KEY = "sk-CCoYJEJcm2mL4YH7uRRw9DPgXQj2f8873F1D98uXtuwclUwW"


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    source: str  # 来源: "rag", "experience", "case"
    content: str  # 内容
    score: float  # 相关性得分
    metadata: Dict[str, Any]  # 元数据
    
    def to_reference_string(self) -> str:
        """转换为参考资料字符串格式"""
        source_name = {
            "rag": "医学指南",
            "experience": "临床经验",
            "case": "相似病例"
        }.get(self.source, self.source)
        
        return f"【{source_name}】{self.content}"


class KnowledgeRetriever:
    """
    统一知识检索服务
    
    整合RAG、经验库、病例库的检索能力
    """
    
    def __init__(
        self,
        enable_rag: bool = True,
        enable_experience: bool = True,
        enable_case: bool = True,
        rag_index_dir: str = "./rag/rag_index",
        memory_db_root: str = "./exp/A-mem-sys/A-mem-sys/memory_db",
        experience_collection: str = "experience_100000",
        case_collection: str = "case_100000",
        api_key: str = API_KEY,
        api_base_url: str = API_BASE_URL,
    ):
        """
        初始化知识检索服务
        
        Args:
            enable_rag: 是否启用RAG指南检索
            enable_experience: 是否启用经验库检索
            enable_case: 是否启用病例库检索
            rag_index_dir: RAG索引目录
            memory_db_root: A-Mem数据库根目录
            experience_collection: 经验库collection名称
            case_collection: 病例库collection名称
        """
        self.enable_rag = enable_rag
        self.enable_experience = enable_experience
        self.enable_case = enable_case
        self.api_key = api_key
        self.api_base_url = api_base_url
        
        # 惰性加载各检索器
        self._rag_retriever = None
        self._experience_retriever = None
        self._case_retriever = None
        
        self.rag_index_dir = rag_index_dir
        self.memory_db_root = memory_db_root
        self.experience_collection = experience_collection
        self.case_collection = case_collection
        
        print(f"[知识检索服务] 初始化完成")
        print(f"  - RAG指南检索: {'启用' if enable_rag else '禁用'}")
        print(f"  - 经验库检索: {'启用' if enable_experience else '禁用'}")
        print(f"  - 病例库检索: {'启用' if enable_case else '禁用'}")
    
    @property
    def rag_retriever(self):
        """惰性加载RAG检索器"""
        if self._rag_retriever is None and self.enable_rag:
            try:
                from hybrid_retriever import HybridLLMRerankRetriever
                
                if os.path.exists(self.rag_index_dir):
                    self._rag_retriever = HybridLLMRerankRetriever(
                        index_dir=self.rag_index_dir,
                        api_key=self.api_key,
                        base_url=self.api_base_url,
                    )
                    print(f"[RAG] 加载成功: {self.rag_index_dir}")
                else:
                    print(f"[RAG] 索引目录不存在: {self.rag_index_dir}")
            except Exception as e:
                print(f"[RAG] 加载失败: {e}")
        return self._rag_retriever
    
    @property
    def experience_retriever(self):
        """惰性加载经验库检索器"""
        if self._experience_retriever is None and self.enable_experience:
            try:
                chroma_path = os.path.join(
                    self.memory_db_root, 
                    f"chroma_{self.experience_collection}"
                )
                if os.path.exists(chroma_path):
                    from chromadb import PersistentClient
                    client = PersistentClient(path=chroma_path)
                    # 尝试获取collection
                    try:
                        self._experience_retriever = client.get_collection(name="memories")
                        print(f"[经验库] 加载成功: {chroma_path}")
                    except Exception:
                        # 尝试其他collection名
                        collections = client.list_collections()
                        if collections:
                            self._experience_retriever = collections[0]
                            print(f"[经验库] 加载成功: {chroma_path} (collection: {collections[0].name})")
                else:
                    print(f"[经验库] 数据目录不存在: {chroma_path}")
            except Exception as e:
                print(f"[经验库] 加载失败: {e}")
        return self._experience_retriever
    
    @property
    def case_retriever(self):
        """惰性加载病例库检索器"""
        if self._case_retriever is None and self.enable_case:
            try:
                chroma_path = os.path.join(
                    self.memory_db_root, 
                    f"chroma_{self.case_collection}"
                )
                if os.path.exists(chroma_path):
                    from chromadb import PersistentClient
                    client = PersistentClient(path=chroma_path)
                    try:
                        self._case_retriever = client.get_collection(name="memories")
                        print(f"[病例库] 加载成功: {chroma_path}")
                    except Exception:
                        collections = client.list_collections()
                        if collections:
                            self._case_retriever = collections[0]
                            print(f"[病例库] 加载成功: {chroma_path} (collection: {collections[0].name})")
                else:
                    print(f"[病例库] 数据目录不存在: {chroma_path}")
            except Exception as e:
                print(f"[病例库] 加载失败: {e}")
        return self._case_retriever
    
    def retrieve_rag(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        检索RAG医学指南
        
        Args:
            query: 检索查询
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        results = []
        if not self.rag_retriever:
            return results
        
        try:
            # 使用混合检索 + LLM重排
            docs_with_scores = self.rag_retriever.search_with_scores(
                query=query, 
                top_k=top_k,
                initial_k=top_k * 2
            )
            
            for score, doc in docs_with_scores:
                results.append(RetrievalResult(
                    source="rag",
                    content=doc.page_content,
                    score=float(score),
                    metadata={
                        "source_file": doc.metadata.get("source", ""),
                        "page": doc.metadata.get("page", ""),
                        "uid": doc.metadata.get("uid", ""),
                    }
                ))
        except Exception as e:
            print(f"[RAG检索] 失败: {e}")
        
        return results
    
    def retrieve_experience(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        检索经验库
        
        Args:
            query: 检索查询
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        results = []
        if not self.experience_retriever:
            return results
        
        try:
            # ChromaDB查询
            search_results = self.experience_retriever.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if search_results and search_results.get("documents"):
                docs = search_results["documents"][0]
                metadatas = search_results.get("metadatas", [[]])[0]
                distances = search_results.get("distances", [[]])[0]
                
                for i, doc in enumerate(docs):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    dist = distances[i] if i < len(distances) else 0
                    # 转换距离为相似度分数
                    score = 1.0 / (1.0 + dist)
                    
                    results.append(RetrievalResult(
                        source="experience",
                        content=doc,
                        score=score,
                        metadata={
                            "context": meta.get("context", ""),
                            "keywords": meta.get("keywords", []),
                            "tags": meta.get("tags", []),
                            "category": meta.get("category", ""),
                        }
                    ))
        except Exception as e:
            print(f"[经验库检索] 失败: {e}")
        
        return results
    
    def retrieve_cases(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        检索相似病例
        
        Args:
            query: 检索查询
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        results = []
        if not self.case_retriever:
            return results
        
        try:
            # ChromaDB查询
            search_results = self.case_retriever.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if search_results and search_results.get("documents"):
                docs = search_results["documents"][0]
                metadatas = search_results.get("metadatas", [[]])[0]
                distances = search_results.get("distances", [[]])[0]
                
                for i, doc in enumerate(docs):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    dist = distances[i] if i < len(distances) else 0
                    score = 1.0 / (1.0 + dist)
                    
                    results.append(RetrievalResult(
                        source="case",
                        content=doc,
                        score=score,
                        metadata={
                            "context": meta.get("context", ""),
                            "keywords": meta.get("keywords", []),
                            "tags": meta.get("tags", []),
                            "diagnosis": meta.get("category", ""),
                        }
                    ))
        except Exception as e:
            print(f"[病例库检索] 失败: {e}")
        
        return results
    
    def retrieve_all(
        self, 
        query: str, 
        rag_k: int = 3, 
        experience_k: int = 3, 
        case_k: int = 3
    ) -> Dict[str, List[RetrievalResult]]:
        """
        从所有来源检索
        
        Args:
            query: 检索查询
            rag_k: RAG返回数量
            experience_k: 经验库返回数量
            case_k: 病例库返回数量
            
        Returns:
            按来源分组的检索结果
        """
        results = {
            "rag": [],
            "experience": [],
            "case": []
        }
        
        if self.enable_rag:
            results["rag"] = self.retrieve_rag(query, rag_k)
            
        if self.enable_experience:
            results["experience"] = self.retrieve_experience(query, experience_k)
            
        if self.enable_case:
            results["case"] = self.retrieve_cases(query, case_k)
        
        return results
    
    def format_reference(
        self,
        results: Dict[str, List[RetrievalResult]],
        include_rag: bool = True,
        include_experience: bool = True,
        include_case: bool = True,
        max_total: int = 10
    ) -> str:
        """
        将检索结果格式化为参考资料字符串
        
        Args:
            results: 检索结果字典
            include_rag: 是否包含RAG结果
            include_experience: 是否包含经验库结果
            include_case: 是否包含病例库结果
            max_total: 最大参考条目数
            
        Returns:
            格式化的参考资料字符串
        """
        all_results = []
        
        if include_rag:
            all_results.extend(results.get("rag", []))
        if include_experience:
            all_results.extend(results.get("experience", []))
        if include_case:
            all_results.extend(results.get("case", []))
        
        if not all_results:
            return "【暂无额外参考资料】"
        
        # 按分数排序，取前max_total个
        all_results.sort(key=lambda x: x.score, reverse=True)
        top_results = all_results[:max_total]
        
        # 分组格式化
        rag_refs = []
        exp_refs = []
        case_refs = []
        
        for r in top_results:
            if r.source == "rag":
                source_file = r.metadata.get("source_file", "")
                filename = os.path.basename(source_file) if source_file else "指南文献"
                rag_refs.append(f"  - [{filename}] {r.content[:300]}...")
            elif r.source == "experience":
                context = r.metadata.get("context", "临床经验")
                exp_refs.append(f"  - [{context}] {r.content[:300]}...")
            elif r.source == "case":
                diag = r.metadata.get("diagnosis", "相似病例")
                case_refs.append(f"  - [{diag}] {r.content[:300]}...")
        
        sections = []
        
        if rag_refs:
            sections.append("【医学指南参考】\n" + "\n".join(rag_refs))
        if exp_refs:
            sections.append("【临床经验参考】\n" + "\n".join(exp_refs))
        if case_refs:
            sections.append("【相似病例参考】\n" + "\n".join(case_refs))
        
        return "\n\n".join(sections) if sections else "【暂无额外参考资料】"
    
    def _extract_retrieval_elements(self, rewritten_text: str, expert_specialty: str) -> str:
        """
        从Step-2专家语义重写输出中提取结构化检索要素
        
        Step-2输出包含:
        A. 医学化重写段（主诉重写、现病史重写、关键阴性线索）
        B. 结构化检索要素摘要（主要症状、时间特征、伴随症状、危险因素等）
        
        提取B部分用于更精准的检索
        """
        import re
        
        # 尝试提取结构化要素部分
        extracted_elements = []
        
        # 1. 提取主要症状
        symptom_patterns = [
            r'主要症状[：:]\s*([^\n]+)',
            r'症状集合[：:]\s*([^\n]+)',
            r'主诉重写[：:]\s*([^\n]+)',
        ]
        for pattern in symptom_patterns:
            match = re.search(pattern, rewritten_text)
            if match:
                extracted_elements.append(match.group(1).strip())
                break
        
        # 2. 提取时间特征
        time_patterns = [
            r'时间特征[：:]\s*([^\n]+)',
            r'病程[：:]\s*([^\n]+)',
            r'起病[：:]\s*([^\n]+)',
        ]
        for pattern in time_patterns:
            match = re.search(pattern, rewritten_text)
            if match:
                extracted_elements.append(match.group(1).strip())
                break
        
        # 3. 提取危险因素/既往史
        risk_patterns = [
            r'危险因素[：:]\s*([^\n]+)',
            r'既往史[：:]\s*([^\n]+)',
            r'背景信息[：:]\s*([^\n]+)',
        ]
        for pattern in risk_patterns:
            match = re.search(pattern, rewritten_text)
            if match:
                extracted_elements.append(match.group(1).strip())
                break
        
        # 4. 提取关键检查结果
        exam_patterns = [
            r'实验室[检查]*[：:]\s*([^\n]+)',
            r'检查[结果]*[：:]\s*([^\n]+)',
            r'影像[学检查]*[：:]\s*([^\n]+)',
        ]
        for pattern in exam_patterns:
            match = re.search(pattern, rewritten_text)
            if match:
                extracted_elements.append(match.group(1).strip())
                break
        
        # 如果成功提取到结构化要素
        if extracted_elements:
            query = f"{expert_specialty} {' '.join(extracted_elements)}"
        else:
            # 回退：使用前500字符 + 专科
            truncated = rewritten_text[:500].replace('\n', ' ')
            query = f"{expert_specialty} {truncated}"
        
        return query
    
    def retrieve_for_expert(
        self,
        rewritten_query: str,
        expert_specialty: str,
        rag_k: int = 3,
        experience_k: int = 3,
        case_k: int = 3
    ) -> str:
        """
        为特定专家检索参考资料
        
        根据专家的Step-2语义重写结果，提取结构化检索要素进行检索。
        不同专家会因视角不同产生不同的检索查询。
        
        Args:
            rewritten_query: 专家语义重写后的完整输出
            expert_specialty: 专家专科
            rag_k: RAG返回数量
            experience_k: 经验库返回数量
            case_k: 病例库返回数量
            
        Returns:
            格式化的参考资料字符串
        """
        # 从Step-2输出中提取结构化检索要素
        enhanced_query = self._extract_retrieval_elements(rewritten_query, expert_specialty)
        
        # 显示实际检索查询（截断显示）
        display_query = enhanced_query[:80] + "..." if len(enhanced_query) > 80 else enhanced_query
        print(f"  [检索查询] {display_query}")
        
        # 检索所有来源
        results = self.retrieve_all(
            query=enhanced_query,
            rag_k=rag_k,
            experience_k=experience_k,
            case_k=case_k
        )
        
        # 统计检索结果
        total_rag = len(results.get("rag", []))
        total_exp = len(results.get("experience", []))
        total_case = len(results.get("case", []))
        print(f"  [检索结果] RAG:{total_rag}, 经验:{total_exp}, 病例:{total_case}")
        
        # 格式化为参考资料
        reference = self.format_reference(results)
        
        return reference


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("知识检索服务测试")
    print("=" * 60)
    
    retriever = KnowledgeRetriever(
        enable_rag=True,
        enable_experience=True,
        enable_case=True,
    )
    
    # 测试查询
    test_query = "腹痛 恶心呕吐 发热 白细胞升高 疑似阑尾炎"
    
    print(f"\n测试查询: {test_query}")
    print("-" * 60)
    
    reference = retriever.retrieve_for_expert(
        rewritten_query=test_query,
        expert_specialty="消化内科",
    )
    
    print("\n参考资料:")
    print(reference)

