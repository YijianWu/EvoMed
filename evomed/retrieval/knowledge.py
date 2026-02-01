# -*- coding: utf-8 -*-
"""
Knowledge Retrieval Service

Integrates three major knowledge sources:
1. RAG Medical Guideline Retrieval (BM25 + FAISS Hybrid Retrieval + LLM Rerank)
2. Experience Library Retrieval (A-Mem System)
3. Case Library Retrieval (Engine System)

Used to provide reference materials for Step-2/Step-3.
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

# API configuration
API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")


@dataclass
class RetrievalResult:
    """Retrieval Result Data Class"""
    source: str  # Source: "rag", "experience", "case"
    content: str  # Content
    score: float  # Relevance score
    metadata: Dict[str, Any]  # Metadata
    
    def to_reference_string(self) -> str:
        """Converts to reference material string format"""
        source_name = {
            "rag": "Medical Guidelines",
            "experience": "Clinical Experience",
            "case": "Similar Cases"
        }.get(self.source, self.source)
        
        return f"【{source_name}】{self.content}"


class KnowledgeRetriever:
    """
    Unified Knowledge Retrieval Service
    
    Integrates retrieval capabilities of RAG, Experience Library, and Case Library.
    """
    
    def __init__(
        self,
        enable_rag: bool = True,
        enable_experience: bool = True,
        enable_case: bool = True,
        rag_index_dir: str = "./rag/rag_index",
        memory_db_root: str = "./exp/repository/memory_db",
        experience_collection: str = "experience_100000",
        case_collection: str = "case_100000",
        api_key: str = API_KEY,
        api_base_url: str = API_BASE_URL,
    ):
        """
        Initializes the knowledge retrieval service
        
        Args:
            enable_rag: Whether to enable RAG guideline retrieval
            enable_experience: Whether to enable experience library retrieval
            enable_case: Whether to enable case library retrieval
            rag_index_dir: RAG index directory
            memory_db_root: A-Mem database root directory
            experience_collection: Experience library collection name
            case_collection: Case library collection name
        """
        self.enable_rag = enable_rag
        self.enable_experience = enable_experience
        self.enable_case = enable_case
        self.api_key = api_key
        self.api_base_url = api_base_url
        
        # Lazy load retrievers
        self._rag_retriever = None
        self._experience_retriever = None
        self._case_retriever = None
        
        self.rag_index_dir = rag_index_dir
        self.memory_db_root = memory_db_root
        self.experience_collection = experience_collection
        self.case_collection = case_collection
        
        print(f"[Knowledge Retrieval Service] Initialization complete")
        print(f"  - RAG Guideline Retrieval: {'Enabled' if enable_rag else 'Disabled'}")
        print(f"  - Experience Library Retrieval: {'Enabled' if enable_experience else 'Disabled'}")
        print(f"  - Case Library Retrieval: {'Enabled' if enable_case else 'Disabled'}")
    
    @property
    def rag_retriever(self):
        """Lazy load RAG retriever"""
        if self._rag_retriever is None and self.enable_rag:
            try:
                from evomed.retrieval.hybrid import HybridLLMRerankRetriever
                
                if os.path.exists(self.rag_index_dir):
                    self._rag_retriever = HybridLLMRerankRetriever(
                        index_dir=self.rag_index_dir,
                        api_key=self.api_key,
                        base_url=self.api_base_url,
                    )
                    print(f"[RAG] Loaded successfully: {self.rag_index_dir}")
                else:
                    print(f"[RAG] Index directory does not exist: {self.rag_index_dir}")
            except Exception as e:
                print(f"[RAG] Failed to load: {e}")
        return self._rag_retriever
    
    @property
    def experience_retriever(self):
        """Lazy load experience library retriever"""
        if self._experience_retriever is None and self.enable_experience:
            try:
                chroma_path = os.path.join(
                    self.memory_db_root, 
                    f"chroma_{self.experience_collection}"
                )
                if os.path.exists(chroma_path):
                    from chromadb import PersistentClient
                    client = PersistentClient(path=chroma_path)
                    # Try to get collection
                    try:
                        self._experience_retriever = client.get_collection(name="memories")
                        print(f"[Experience Library] Loaded successfully: {chroma_path}")
                    except Exception:
                        # Try other collection names
                        collections = client.list_collections()
                        if collections:
                            self._experience_retriever = collections[0]
                            print(f"[Experience Library] Loaded successfully: {chroma_path} (collection: {collections[0].name})")
                else:
                    print(f"[Experience Library] Data directory does not exist: {chroma_path}")
            except Exception as e:
                print(f"[Experience Library] Failed to load: {e}")
        return self._experience_retriever
    
    @property
    def case_retriever(self):
        """Lazy load case library retriever"""
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
                        print(f"[Case Library] Loaded successfully: {chroma_path}")
                    except Exception:
                        collections = client.list_collections()
                        if collections:
                            self._case_retriever = collections[0]
                            print(f"[Case Library] Loaded successfully: {chroma_path} (collection: {collections[0].name})")
                else:
                    print(f"[Case Library] Data directory does not exist: {chroma_path}")
            except Exception as e:
                print(f"[Case Library] Failed to load: {e}")
        return self._case_retriever
    
    def retrieve_rag(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieves RAG medical guidelines
        
        Args:
            query: Retrieval query
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        results = []
        if not self.rag_retriever:
            return results
        
        try:
            # Use hybrid retrieval + LLM rerank
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
            print(f"[RAG Retrieval] Failed: {e}")
        
        return results
    
    def retrieve_experience(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieves from experience library
        
        Args:
            query: Retrieval query
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        results = []
        if not self.experience_retriever:
            return results
        
        try:
            # ChromaDB query
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
                    # Convert distance to similarity score
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
            print(f"[Experience Library Retrieval] Failed: {e}")
        
        return results
    
    def retrieve_cases(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieves similar cases
        
        Args:
            query: Retrieval query
            top_k: Number of results to return
            
        Returns:
            List of retrieval results
        """
        results = []
        if not self.case_retriever:
            return results
        
        try:
            # ChromaDB query
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
            print(f"[Case Library Retrieval] Failed: {e}")
        
        return results
    
    def retrieve_all(
        self, 
        query: str, 
        rag_k: int = 3, 
        experience_k: int = 3, 
        case_k: int = 3
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieves from all sources
        
        Args:
            query: Retrieval query
            rag_k: RAG return count
            experience_k: Experience library return count
            case_k: Case library return count
            
        Returns:
            Retrieval results grouped by source
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
        Formats retrieval results into a reference material string
        
        Args:
            results: Retrieval results dictionary
            include_rag: Whether to include RAG results
            include_experience: Whether to include experience library results
            include_case: Whether to include case library results
            max_total: Maximum number of reference entries
            
        Returns:
            Formatted reference material string
        """
        all_results = []
        
        if include_rag:
            all_results.extend(results.get("rag", []))
        if include_experience:
            all_results.extend(results.get("experience", []))
        if include_case:
            all_results.extend(results.get("case", []))
        
        if not all_results:
            return "【No additional reference materials available】"
        
        # Sort by score and take top max_total
        all_results.sort(key=lambda x: x.score, reverse=True)
        top_results = all_results[:max_total]
        
        # Group and format
        rag_refs = []
        exp_refs = []
        case_refs = []
        
        for r in top_results:
            if r.source == "rag":
                source_file = r.metadata.get("source_file", "")
                filename = os.path.basename(source_file) if source_file else "Guideline Document"
                rag_refs.append(f"  - [{filename}] {r.content[:300]}...")
            elif r.source == "experience":
                context = r.metadata.get("context", "Clinical Experience")
                exp_refs.append(f"  - [{context}] {r.content[:300]}...")
            elif r.source == "case":
                diag = r.metadata.get("diagnosis", "Similar Case")
                case_refs.append(f"  - [{diag}] {r.content[:300]}...")
        
        sections = []
        
        if rag_refs:
            sections.append("【Medical Guideline References】\n" + "\n".join(rag_refs))
        if exp_refs:
            sections.append("【Clinical Experience References】\n" + "\n".join(exp_refs))
        if case_refs:
            sections.append("【Similar Case References】\n" + "\n".join(case_refs))
        
        return "\n\n".join(sections) if sections else "【No additional reference materials available】"
    
    def _extract_retrieval_elements(self, rewritten_text: str, expert_specialty: str) -> str:
        """
        Extracts structured retrieval elements from Step-2 expert semantic rewrite output
        
        Step-2 output includes:
        A. Medicalized rewriting segment (chief complaint rewrite, HPI rewrite, key negative clues)
        B. Structured retrieval element summary (main symptoms, temporal characteristics, accompanying symptoms, risk factors, etc.)
        
        Extracting Part B for more precise retrieval.
        """
        import re
        
        # Try to extract structured elements section
        extracted_elements = []
        
        # 1. Extract main symptoms
        symptom_patterns = [
            r'Main Symptoms[：:]\s*([^\n]+)',
            r'Symptom Collection[：:]\s*([^\n]+)',
            r'Chief Complaint Rewrite[：:]\s*([^\n]+)',
        ]
        for pattern in symptom_patterns:
            match = re.search(pattern, rewritten_text, re.IGNORECASE)
            if match:
                extracted_elements.append(match.group(1).strip())
                break
        
        # 2. Extract temporal characteristics
        time_patterns = [
            r'Temporal Characteristics[：:]\s*([^\n]+)',
            r'Course of Illness[：:]\s*([^\n]+)',
            r'Onset[：:]\s*([^\n]+)',
        ]
        for pattern in time_patterns:
            match = re.search(pattern, rewritten_text, re.IGNORECASE)
            if match:
                extracted_elements.append(match.group(1).strip())
                break
        
        # 3. Extract risk factors/past history
        risk_patterns = [
            r'Risk Factors[：:]\s*([^\n]+)',
            r'Past History[：:]\s*([^\n]+)',
            r'Background Info[：:]\s*([^\n]+)',
        ]
        for pattern in risk_patterns:
            match = re.search(pattern, rewritten_text, re.IGNORECASE)
            if match:
                extracted_elements.append(match.group(1).strip())
                break
        
        # 4. Extract key examination results
        exam_patterns = [
            r'Laboratory [Results]*[：:]\s*([^\n]+)',
            r'Examination [Results]*[：:]\s*([^\n]+)',
            r'Imaging [Results]*[：:]\s*([^\n]+)',
        ]
        for pattern in exam_patterns:
            match = re.search(pattern, rewritten_text, re.IGNORECASE)
            if match:
                extracted_elements.append(match.group(1).strip())
                break
        
        # If successfully extracted structured elements
        if extracted_elements:
            query = f"{expert_specialty} {' '.join(extracted_elements)}"
        else:
            # Fallback: use first 500 characters + specialty
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
        Retrieves reference materials for a specific expert
        
        Extracts structured retrieval elements from the expert's Step-2 semantic rewrite results for retrieval.
        Different experts generate different retrieval queries due to their unique perspectives.
        
        Args:
            rewritten_query: Complete output after expert semantic rewrite
            expert_specialty: Expert specialty
            rag_k: RAG return count
            experience_k: Experience library return count
            case_k: Case library return count
            
        Returns:
            Formatted reference material string
        """
        # Extract structured retrieval elements from Step-2 output
        enhanced_query = self._extract_retrieval_elements(rewritten_query, expert_specialty)
        
        # Show actual retrieval query (truncated display)
        display_query = enhanced_query[:80] + "..." if len(enhanced_query) > 80 else enhanced_query
        print(f"  [Retrieval Query] {display_query}")
        
        # Retrieve from all sources
        results = self.retrieve_all(
            query=enhanced_query,
            rag_k=rag_k,
            experience_k=experience_k,
            case_k=case_k
        )
        
        # Stat retrieval results
        total_rag = len(results.get("rag", []))
        total_exp = len(results.get("experience", []))
        total_case = len(results.get("case", []))
        print(f"  [Retrieval Results] RAG:{total_rag}, Experience:{total_exp}, Case:{total_case}")
        
        # Format as reference materials
        reference = self.format_reference(results)
        
        return reference


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Knowledge Retrieval Service Test")
    print("=" * 60)
    
    retriever = KnowledgeRetriever(
        enable_rag=True,
        enable_experience=True,
        enable_case=True,
    )
    
    # Test query
    test_query = "Abdominal pain, nausea and vomiting, fever, elevated WBC, suspected appendicitis"
    
    print(f"\nTest Query: {test_query}")
    print("-" * 60)
    
    reference = retriever.retrieve_for_expert(
        rewritten_query=test_query,
        expert_specialty="Gastroenterology",
    )
    
    print("\nReference Materials:")
    print(reference)
