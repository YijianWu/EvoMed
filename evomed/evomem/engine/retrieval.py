"""Retrieval logic for Engine."""

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
    """Retrieval Result"""
    bullet_id: str
    content: str
    score: float
    source: str = "experience"


class SemanticRetriever:
    """
    Semantic retriever based on vector embeddings
    Uses SentenceTransformer for text vectorization, providing better semantic matching
    """

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2", lazy_load: bool = True):
        """
        Initialize semantic retriever

        Args:
            model_name: SentenceTransformer model name, using lightweight model for speed
            lazy_load: Whether to lazy load model (default True, loads on first use)
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

        # FAISS index related
        self.faiss_index = None
        self.index_bullet_ids = []  # Maintain mapping between IDs and vectors
        self.embedding_dim = None  # Vector dimension
        self.index_needs_update = False  # Flag whether index needs update
        self.index_lock = Lock()  # Thread lock protecting FAISS index

    def _build_faiss_index(self):
        """Build FAISS index"""
        if not FAISS_AVAILABLE or not self.content_cache:
            return

        try:
            # Get all embeddings
            embeddings = []
            bullet_ids = []

            for bullet_id, content in self.content_cache.items():
                emb = self._get_embedding(content)
                if emb is not None:
                    embeddings.append(emb)
                    bullet_ids.append(bullet_id)

            if not embeddings:
                return

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.embedding_dim = embeddings_array.shape[1]

            # L2 normalization, making inner product equal to cosine similarity (range 0-1)
            faiss.normalize_L2(embeddings_array)

            # Create FAISS index
            if self.embedding_dim <= 768:  # For medium dimensions, use IndexIVFFlat
                nlist = min(100, max(4, len(embeddings) // 39))  # IVF parameter
                quantizer = faiss.IndexFlatIP(self.embedding_dim)  # Inner product index
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                # Train index
                self.faiss_index.train(embeddings_array)
            else:
                # For high dimensions, use IndexFlatIP (brute force but memory friendly)
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            # Add vectors to index
            self.faiss_index.add(embeddings_array)
            self.index_bullet_ids = bullet_ids

            print(f"[INFO] Built FAISS index with {len(bullet_ids)} vectors, dimension: {self.embedding_dim}")

        except Exception as e:
            print(f"[WARN] Failed to build FAISS index: {e}, falling back to brute force")
            self.faiss_index = None

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index"""
        self.faiss_index = None
        self.index_bullet_ids = []
        self._build_faiss_index()

    def _get_embedding(self, text: str):
        """Get vector embedding of text (with cache)"""
        if text not in self.embeddings_cache:
            # Lazy load model
            if self.model is None and self.lazy_load and SEMANTIC_DEPS_AVAILABLE:
                try:
                    print(f"[INFO] Lazy loading SentenceTransformer model: {self.model_name}")
                    self.model = SentenceTransformer(self.model_name)
                except Exception as e:
                    print(f"[WARN] Failed to lazy load model: {e}")
                    self.lazy_load = False  # Stop trying to load
            
            if self.model is not None and SEMANTIC_DEPS_AVAILABLE:
                # Clean text, retain key information
                clean_text = self._preprocess_text(text)
                self.embeddings_cache[text] = self.model.encode([clean_text])[0]
            else:
                # fallback: use simple numerical representation (for string similarity calculation)
                self.embeddings_cache[text] = hash(text) % 1000  # Simple numerical representation
        return self.embeddings_cache[text]

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to improve retrieval quality"""
        # Retain diagnostic keywords, filter noise
        text = text.lower()

        # Extract key patterns: disease names, symptom descriptions
        import re

        # Retain Chinese characters, English words, numbers
        clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        # Compress consecutive spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def add_experience(self, bullet_id: str, content: str):
        """Add experience entry to retrieval index"""
        self.content_cache[bullet_id] = content
        # Precompute embedding
        self._get_embedding(content)
        # Mark FAISS index as needing update
        self.index_needs_update = True

    def search_similar(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve most relevant experiences based on semantic similarity

        Args:
            query: Retrieval query
            top_k: Number of results to return

        Returns:
            List of retrieval results, sorted by similarity
        """
        if not self.content_cache:
            return []

        if self.model is None:
            # Fallback to simple string matching
            return self._fallback_search(query, top_k)

        if self.model is not None and SEMANTIC_DEPS_AVAILABLE:
            try:
                # Check and rebuild FAISS index (if needed) - protected by thread lock
                with self.index_lock:
                    if self.index_needs_update or self.faiss_index is None:
                        # print("[INFO] Rebuilding FAISS index for updated experience base...")
                        self._rebuild_faiss_index()
                        self.index_needs_update = False

                # Prioritize using FAISS for efficient retrieval
                if self.faiss_index is not None and FAISS_AVAILABLE:
                    return self._faiss_search(query, top_k)
                else:
                    # Fall back to brute force search
                    return self._brute_force_search(query, top_k)

            except Exception as e:
                print(f"[WARN] Semantic search failed: {e}, falling back to string matching")
                return self._fallback_search(query, top_k)
        else:
            # No semantic dependencies, use string matching directly
            return self._fallback_search(query, top_k)

    def _faiss_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Use FAISS for efficient similarity retrieval"""
        try:
            # Get query embedding
            query_emb = self._get_embedding(query)
            if query_emb is None:
                return []

            query_array = np.array([query_emb], dtype=np.float32)
            faiss.normalize_L2(query_array)  # Normalize query vector

            # FAISS search - protected by thread lock
            with self.index_lock:
                if self.faiss_index is None:
                    print("[WARN] FAISS index is None, falling back to brute force")
                    return self._brute_force_search(query, top_k)

                scores, indices = self.faiss_index.search(query_array, min(top_k, len(self.index_bullet_ids)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.index_bullet_ids):  # Valid index
                    bullet_id = self.index_bullet_ids[idx]
                    content = self.content_cache.get(bullet_id, "")

                    results.append(RetrievalResult(
                        bullet_id=bullet_id,
                        content=content,
                        score=float(score),  # FAISS returns inner product score
                        source="experience"
                    ))

            # Sort by score descending (FAISS might not guarantee full sorting)
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        except Exception as e:
            print(f"[WARN] FAISS search failed: {e}, falling back to brute force")
            return self._brute_force_search(query, top_k)

    def _brute_force_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Brute force search method (original implementation)"""
        try:
            # Get embedding of query
            query_emb = self._get_embedding(query)

            results = []
            for bullet_id, content in self.content_cache.items():
                content_emb = self._get_embedding(content)

                # Calculate cosine similarity
                similarity = cosine_similarity([query_emb], [content_emb])[0][0]

                results.append(RetrievalResult(
                    bullet_id=bullet_id,
                    content=content,
                    score=float(similarity),
                    source="experience"
                ))

            # Sort by similarity descending
            results.sort(key=lambda x: x.score, reverse=True)

            # Return top_k results
            return results[:top_k]

        except Exception as e:
            print(f"[WARN] Brute force search failed: {e}, falling back to string matching")
            return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Fallback scheme: use simple string similarity matching"""
        results = []
        query_lower = query.lower()

        for bullet_id, content in self.content_cache.items():
            content_lower = content.lower()
            similarity = SequenceMatcher(None, query_lower, content_lower).ratio()

            if similarity > 0.5:  # Similarity threshold - increase quality assurance
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
        Check if new content duplicates existing experience (FAISS optimized version)

        Args:
            new_content: New experience content
            threshold: Duplication threshold

        Returns:
            (is_duplicate, ID of most similar entry)
        """
        if not self.content_cache:
            return False, ""

        # Check and rebuild FAISS index (if needed) - protected by thread lock
        with self.index_lock:
            if self.index_needs_update or self.faiss_index is None:
                self._rebuild_faiss_index()
                self.index_needs_update = False

        # Prioritize using FAISS for efficient duplication check
        if self.faiss_index is not None and FAISS_AVAILABLE and self.model is not None and SEMANTIC_DEPS_AVAILABLE:
            try:
                new_emb = self._get_embedding(new_content)
                if new_emb is None:
                    return False, ""

                query_array = np.array([new_emb], dtype=np.float32)
                faiss.normalize_L2(query_array)  # Normalize query vector

                # Search for most similar 1 result - protected by thread lock
                with self.index_lock:
                    scores, indices = self.faiss_index.search(query_array, 1)

                if indices[0][0] < len(self.index_bullet_ids):
                    best_score = float(scores[0][0])
                    best_bullet_id = self.index_bullet_ids[indices[0][0]]
                    return best_score >= threshold, best_bullet_id

            except Exception as e:
                print(f"[WARN] FAISS duplicate check failed: {e}, falling back to brute force")

        # Fall back to brute force check
        return self._check_duplicate_brute_force(new_content, threshold)

    def _check_duplicate_brute_force(self, new_content: str, threshold: float = 0.8) -> Tuple[bool, str]:
        """Brute force duplicate check (original implementation)"""
        new_content_clean = self._preprocess_text(new_content)
        best_score = 0.0
        best_bullet_id = ""

        for bullet_id, existing_content in self.content_cache.items():
            existing_clean = self._preprocess_text(existing_content)

            # Calculate similarity
            if self.model is not None and SEMANTIC_DEPS_AVAILABLE:
                # Use semantic similarity
                new_emb = self._get_embedding(new_content)
                existing_emb = self._get_embedding(existing_content)
                similarity = cosine_similarity([new_emb], [existing_emb])[0][0]
            else:
                # Use string similarity
                similarity = SequenceMatcher(None, new_content_clean, existing_clean).ratio()

            if similarity > best_score:
                best_score = similarity
                best_bullet_id = bullet_id

        return best_score >= threshold, best_bullet_id


# ------------------------------------------------------------------ #
# Modular Semantic Retriever
# ------------------------------------------------------------------ #

@dataclass
class ModularRetrievalResult:
    """Modular Retrieval Result"""
    bullet_id: str
    fixed_modules: Dict  # Fixed modules (matching basis)
    mutable_modules: Dict  # Iterative modules (can be modified in reflection phase)
    score: float
    section: str = ""
    source: str = "modular_experience"


class ModularSemanticRetriever:
    """
    Modular Semantic Retriever
    
    Core Features:
    1. Build vector index based ONLY on fixed modules (contextual_states + decision_behaviors)
    2. Return full modular experience (including iterative modules) during retrieval
    3. Support updating iterative modules (uncertainty + delayed_assumptions) in reflection phase
    """

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2", lazy_load: bool = True):
        """
        Initialize modular semantic retriever

        Args:
            model_name: SentenceTransformer model name
            lazy_load: Whether to lazy load model
        """
        self.model_name = model_name
        self.lazy_load = lazy_load
        self.model = None
        
        if SEMANTIC_DEPS_AVAILABLE and not lazy_load:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"[INFO] Initialized modular semantic retriever with model: {model_name}")
            except Exception as e:
                print(f"[WARN] Failed to load SentenceTransformer model: {e}")
                print("[WARN] Falling back to simple string matching")

        # Vector cache (only stores vectors of fixed modules)
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        # Fixed module text cache (for vectorization)
        self.fixed_text_cache: Dict[str, str] = {}
        # Full module cache (including iterative modules)
        self.full_modules_cache: Dict[str, Dict] = {}
        # Section mapping
        self.section_cache: Dict[str, str] = {}

        # FAISS index
        self.faiss_index = None
        self.index_bullet_ids: List[str] = []
        self.embedding_dim = None
        self.index_needs_update = False
        self.index_lock = Lock()

    def _ensure_model_loaded(self):
        """Ensure model is loaded"""
        if self.model is None and self.lazy_load and SEMANTIC_DEPS_AVAILABLE:
            try:
                print(f"[INFO] Lazy loading SentenceTransformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[WARN] Failed to lazy load model: {e}")
                self.lazy_load = False

    def add_modular_experience(
        self,
        bullet_id: str,
        section: str,
        fixed_modules: Dict,
        mutable_modules: Dict,
    ):
        """
        Add modular experience to retrieval index

        Args:
            bullet_id: Experience ID
            section: Section name
            fixed_modules: Fixed modules (for vector retrieval)
            mutable_modules: Iterative modules (can be modified in reflection phase)
        """
        # Generate vectorized text from fixed modules
        vector_text = self._build_vector_text(fixed_modules)
        
        # Cache
        self.fixed_text_cache[bullet_id] = vector_text
        self.full_modules_cache[bullet_id] = {
            "fixed_modules": fixed_modules,
            "mutable_modules": mutable_modules,
        }
        self.section_cache[bullet_id] = section
        
        # Precompute embedding
        self._get_embedding(vector_text)
        
        # Mark index as needing update
        self.index_needs_update = True

    def _build_vector_text(self, fixed_modules: Dict) -> str:
        """Build text for vectorization from fixed modules"""
        parts = []
        
        # contextual_states
        cs = fixed_modules.get("contextual_states", {})
        if isinstance(cs, dict):
            if cs.get("scenario"):
                parts.append(f"Scenario: {cs['scenario']}")
            if cs.get("chief_complaint"):
                parts.append(f"Chief Complaint: {cs['chief_complaint']}")
            if cs.get("core_symptoms"):
                parts.append(f"Core Symptoms: {cs['core_symptoms']}")
        
        # decision_behaviors
        db = fixed_modules.get("decision_behaviors", {})
        if isinstance(db, dict):
            if db.get("diagnostic_path"):
                parts.append(f"Diagnostic Path: {db['diagnostic_path']}")
        
        return " | ".join(parts) if parts else ""

    def update_mutable_modules(self, bullet_id: str, mutable_modules: Dict):
        """
        Update iterative modules (does not affect vector index)

        Args:
            bullet_id: Experience ID
            mutable_modules: New iterative module data
        """
        if bullet_id not in self.full_modules_cache:
            print(f"[WARN] Bullet {bullet_id} not found in cache")
            return False
        
        # Only update iterative modules, fixed modules remain unchanged
        self.full_modules_cache[bullet_id]["mutable_modules"] = mutable_modules
        return True

    def _get_embedding(self, text: str):
        """Get vector embedding of text (with cache)"""
        if text not in self.embeddings_cache:
            self._ensure_model_loaded()
            
            if self.model is not None and SEMANTIC_DEPS_AVAILABLE:
                clean_text = self._preprocess_text(text)
                self.embeddings_cache[text] = self.model.encode([clean_text])[0]
            else:
                self.embeddings_cache[text] = hash(text) % 1000
        return self.embeddings_cache[text]

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text"""
        import re
        text = text.lower()
        clean_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text

    def _build_faiss_index(self):
        """Build FAISS index (based ONLY on fixed modules)"""
        if not FAISS_AVAILABLE or not self.fixed_text_cache:
            return

        try:
            embeddings = []
            bullet_ids = []

            for bullet_id, fixed_text in self.fixed_text_cache.items():
                emb = self._get_embedding(fixed_text)
                if emb is not None and not isinstance(emb, int):
                    embeddings.append(emb)
                    bullet_ids.append(bullet_id)

            if not embeddings:
                return

            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.embedding_dim = embeddings_array.shape[1]

            # L2 normalization, making inner product equal to cosine similarity (range 0-1)
            faiss.normalize_L2(embeddings_array)

            if self.embedding_dim <= 768:
                nlist = min(100, max(4, len(embeddings) // 39))
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(
                    quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
                )
                self.faiss_index.train(embeddings_array)
            else:
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            self.faiss_index.add(embeddings_array)
            self.index_bullet_ids = bullet_ids

            print(f"[INFO] Built modular FAISS index with {len(bullet_ids)} vectors")

        except Exception as e:
            print(f"[WARN] Failed to build modular FAISS index: {e}")
            self.faiss_index = None

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index"""
        self.faiss_index = None
        self.index_bullet_ids = []
        self._build_faiss_index()

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[ModularRetrievalResult]:
        """
        Semantic similarity retrieval based on fixed modules

        Args:
            query: Retrieval query (usually diagnosis or case description)
            top_k: Number of results to return
            threshold: Similarity threshold

        Returns:
            List of modular retrieval results (including fixed and iterative modules)
        """
        if not self.fixed_text_cache:
            return []

        self._ensure_model_loaded()

        if self.model is None:
            return self._fallback_search(query, top_k, threshold)

        try:
            with self.index_lock:
                if self.index_needs_update or self.faiss_index is None:
                    self._rebuild_faiss_index()
                    self.index_needs_update = False

            if self.faiss_index is not None and FAISS_AVAILABLE:
                return self._faiss_search(query, top_k, threshold)
            else:
                return self._brute_force_search(query, top_k, threshold)

        except Exception as e:
            print(f"[WARN] Modular semantic search failed: {e}")
            return self._fallback_search(query, top_k, threshold)

    def _faiss_search(
        self, query: str, top_k: int, threshold: float
    ) -> List[ModularRetrievalResult]:
        """Use FAISS for efficient retrieval"""
        try:
            query_emb = self._get_embedding(query)
            if query_emb is None or isinstance(query_emb, int):
                return []

            query_array = np.array([query_emb], dtype=np.float32)
            faiss.normalize_L2(query_array)  # Normalize query vector

            with self.index_lock:
                if self.faiss_index is None:
                    return self._brute_force_search(query, top_k, threshold)
                scores, indices = self.faiss_index.search(
                    query_array, min(top_k, len(self.index_bullet_ids))
                )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.index_bullet_ids) and score >= threshold:
                    bullet_id = self.index_bullet_ids[idx]
                    full_modules = self.full_modules_cache.get(bullet_id, {})
                    
                    results.append(ModularRetrievalResult(
                        bullet_id=bullet_id,
                        fixed_modules=full_modules.get("fixed_modules", {}),
                        mutable_modules=full_modules.get("mutable_modules", {}),
                        score=float(score),
                        section=self.section_cache.get(bullet_id, ""),
                        source="modular_experience",
                    ))

            results.sort(key=lambda x: x.score, reverse=True)
            return results

        except Exception as e:
            print(f"[WARN] FAISS search failed: {e}")
            return self._brute_force_search(query, top_k, threshold)

    def _brute_force_search(
        self, query: str, top_k: int, threshold: float
    ) -> List[ModularRetrievalResult]:
        """Brute force search method"""
        try:
            query_emb = self._get_embedding(query)
            if query_emb is None or isinstance(query_emb, int):
                return self._fallback_search(query, top_k, threshold)

            results = []
            for bullet_id, fixed_text in self.fixed_text_cache.items():
                content_emb = self._get_embedding(fixed_text)
                if isinstance(content_emb, int):
                    continue

                similarity = cosine_similarity([query_emb], [content_emb])[0][0]

                if similarity >= threshold:
                    full_modules = self.full_modules_cache.get(bullet_id, {})
                    results.append(ModularRetrievalResult(
                        bullet_id=bullet_id,
                        fixed_modules=full_modules.get("fixed_modules", {}),
                        mutable_modules=full_modules.get("mutable_modules", {}),
                        score=float(similarity),
                        section=self.section_cache.get(bullet_id, ""),
                        source="modular_experience",
                    ))

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            print(f"[WARN] Brute force search failed: {e}")
            return self._fallback_search(query, top_k, threshold)

    def _fallback_search(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[ModularRetrievalResult]:
        """Fallback scheme: string similarity matching"""
        results = []
        query_lower = query.lower()

        for bullet_id, fixed_text in self.fixed_text_cache.items():
            fixed_lower = fixed_text.lower()
            similarity = SequenceMatcher(None, query_lower, fixed_lower).ratio()

            if similarity >= max(threshold, 0.3):
                full_modules = self.full_modules_cache.get(bullet_id, {})
                results.append(ModularRetrievalResult(
                    bullet_id=bullet_id,
                    fixed_modules=full_modules.get("fixed_modules", {}),
                    mutable_modules=full_modules.get("mutable_modules", {}),
                    score=float(similarity),
                    section=self.section_cache.get(bullet_id, ""),
                    source="modular_experience",
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def check_duplicate(
        self, fixed_modules: Dict, threshold: float = 0.8
    ) -> Tuple[bool, str]:
        """
        Check for duplication with existing experience (based ONLY on fixed modules)

        Args:
            fixed_modules: Fixed modules
            threshold: Duplication threshold

        Returns:
            (is_duplicate, ID of most similar entry)
        """
        if not self.fixed_text_cache:
            return False, ""

        new_text = self._build_vector_text(fixed_modules)
        if not new_text:
            return False, ""

        # Use FAISS for fast check
        with self.index_lock:
            if self.index_needs_update or self.faiss_index is None:
                self._rebuild_faiss_index()
                self.index_needs_update = False

        if self.faiss_index is not None and FAISS_AVAILABLE and self.model is not None:
            try:
                new_emb = self._get_embedding(new_text)
                if new_emb is None or isinstance(new_emb, int):
                    return False, ""

                query_array = np.array([new_emb], dtype=np.float32)
                faiss.normalize_L2(query_array)  # Normalize query vector

                with self.index_lock:
                    scores, indices = self.faiss_index.search(query_array, 1)

                if indices[0][0] < len(self.index_bullet_ids):
                    best_score = float(scores[0][0])
                    best_bullet_id = self.index_bullet_ids[indices[0][0]]
                    return best_score >= threshold, best_bullet_id

            except Exception as e:
                print(f"[WARN] FAISS duplicate check failed: {e}")

        # Fall back to brute force check
        return self._check_duplicate_brute_force(new_text, threshold)

    def _check_duplicate_brute_force(
        self, new_text: str, threshold: float = 0.8
    ) -> Tuple[bool, str]:
        """Brute force duplicate check"""
        new_clean = self._preprocess_text(new_text)
        best_score = 0.0
        best_bullet_id = ""

        for bullet_id, existing_text in self.fixed_text_cache.items():
            existing_clean = self._preprocess_text(existing_text)

            if self.model is not None and SEMANTIC_DEPS_AVAILABLE:
                new_emb = self._get_embedding(new_text)
                existing_emb = self._get_embedding(existing_text)
                if not isinstance(new_emb, int) and not isinstance(existing_emb, int):
                    similarity = cosine_similarity([new_emb], [existing_emb])[0][0]
                else:
                    similarity = SequenceMatcher(None, new_clean, existing_clean).ratio()
            else:
                similarity = SequenceMatcher(None, new_clean, existing_clean).ratio()

            if similarity > best_score:
                best_score = similarity
                best_bullet_id = bullet_id

        return best_score >= threshold, best_bullet_id

    def get_mutable_modules(self, bullet_id: str) -> Optional[Dict]:
        """Get iterative modules of specified experience"""
        full_modules = self.full_modules_cache.get(bullet_id)
        if full_modules:
            return full_modules.get("mutable_modules")
        return None

    def get_full_modules(self, bullet_id: str) -> Optional[Dict]:
        """Get full modules of specified experience"""
        return self.full_modules_cache.get(bullet_id)
