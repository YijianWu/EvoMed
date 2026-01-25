from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import pickle
from nltk.tokenize import word_tokenize
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def simple_tokenize(text):
    return word_tokenize(text)

class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize ChromaDB retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        path_env = os.getenv("AMEM_CHROMA_PATH")
        if path_env:
            self.client = chromadb.PersistentClient(
                path=path_env,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
        else:
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False, allow_reset=True)
            )

        # ✅ Self-maintained SentenceTransformer model for batch + multi-threaded embedding
        self.model_name = model_name
        self.st_model = SentenceTransformer(model_name)

        # ✅ Manually handle embeddings instead of letting Chroma call embedding_function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )

    def add_documents_batch(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
        batch_size: int = 256,
        max_workers: int = 64,
    ):
        """
        Batch add documents to Chroma:
        1) Construct enhanced_document and processed_metadata for each document
        2) Use 64 threads to parallelize SentenceTransformer.encode on multiple chunks
        3) Call self.collection.add(embeddings=..., metadatas=..., ids=...) in the main thread
        """
        if not documents:
            return

        assert len(documents) == len(metadatas) == len(ids), "Length of documents / metadatas / ids must be consistent"

        # 1️⃣ Construct enhanced_document + processed_metadata
        enhanced_docs: List[str] = []
        processed_metas: List[Dict] = []
        new_ids: List[str] = []

        for doc, metadata, doc_id in zip(documents, metadatas, ids):
            enhanced_document = doc

            # Maintain enhancement logic: concatenate context / keywords / tags into text
            if 'context' in metadata and metadata['context'] != "General":
                enhanced_document += f" context: {metadata['context']}"
            
            if 'keywords' in metadata and metadata['keywords']:
                keywords = metadata['keywords'] if isinstance(metadata['keywords'], list) else json.loads(metadata['keywords'])
                if keywords:
                    enhanced_document += f" keywords: {', '.join(keywords)}"
            
            if 'tags' in metadata and metadata['tags']:
                tags = metadata['tags'] if isinstance(metadata['tags'], list) else json.loads(metadata['tags'])
                if tags:
                    enhanced_document += f" tags: {', '.join(tags)}"

            # Convert to serializable metadata
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, list) or isinstance(value, dict):
                    processed_metadata[key] = json.dumps(value, ensure_ascii=False)
                else:
                    processed_metadata[key] = str(value)
            processed_metadata["enhanced_content"] = enhanced_document

            enhanced_docs.append(enhanced_document)
            processed_metas.append(processed_metadata)
            new_ids.append(doc_id)

        # 2️⃣ Chunk by batch_size for parallel embedding
        indices = list(range(0, len(enhanced_docs), batch_size))

        def _encode_chunk(start_idx: int):
            docs_chunk = enhanced_docs[start_idx:start_idx + batch_size]
            metas_chunk = processed_metas[start_idx:start_idx + batch_size]
            ids_chunk = new_ids[start_idx:start_idx + batch_size]

            embeddings = self.st_model.encode(
                docs_chunk,
                batch_size=64,  # Internal batch size for SentenceTransformer
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return docs_chunk, metas_chunk, ids_chunk, embeddings

        # 3️⃣ 64-thread parallel embedding calculation
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_encode_chunk, start_idx): start_idx
                for start_idx in indices
            }
            for future in as_completed(future_to_idx):
                docs_chunk, metas_chunk, ids_chunk, embeddings = future.result()
                results.append((docs_chunk, metas_chunk, ids_chunk, embeddings))

        # 4️⃣ Synchronized write to Chroma in main thread (to avoid multi-threaded DB writes)
        for docs_chunk, metas_chunk, ids_chunk, embeddings in results:
            self.collection.add(
                documents=docs_chunk,
                embeddings=embeddings.tolist(),
                metadatas=metas_chunk,
                ids=ids_chunk,
            )


    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """
        Add a single document: compatibility wrapper using batch logic.
        """
        self.add_documents_batch(
            documents=[document],
            metadatas=[metadata],
            ids=[doc_id],
            batch_size=1,      # Single batch for single document
            max_workers=1      # No need for threads for single document
        )

        
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])
        
    def search(self, query: str, k: int = 5):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        # ✅ Pre-calculate query embedding
        q_emb = self.st_model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        results = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=k
        )
        
        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            for i in range(len(results['metadatas'])):
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    pass
        
        return results
