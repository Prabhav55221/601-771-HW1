"""FAISS-based information retrieval system.

Author: Prabhav Singh
"""
import faiss
import numpy as np
from typing import List, Tuple


class FAISSRetriever:
    """FAISS-based IR system for document retrieval."""
    
    def __init__(self):
        """Initialize FAISS retriever."""
        self.index = None
        self.evidence_doc_ids = None
        
    def build_index(self, evidence_embeddings: np.ndarray, evidence_doc_ids: List[int]):
        """Build FAISS index from evidence embeddings.
        
        Args:
            evidence_embeddings: (n_docs, embed_dim) normalized embeddings
            evidence_doc_ids: List of doc_ids corresponding to embeddings
        """
        print("Building FAISS index...")
        
        # Store doc_ids for retrieval
        self.evidence_doc_ids = evidence_doc_ids
        
        # Create FAISS index for cosine similarity (inner product on normalized vectors)
        embed_dim = evidence_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embed_dim)
        
        # Add embeddings to index
        self.index.add(evidence_embeddings)
        
        print(f"Built FAISS index with {self.index.ntotal} documents")
        
    def search(self, query_embeddings: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-k most similar documents for each query.
        
        Args:
            query_embeddings: (n_queries, embed_dim) normalized query embeddings
            k: Number of top documents to retrieve
            
        Returns:
            similarities: (n_queries, k) similarity scores
            doc_ids: (n_queries, k) retrieved document IDs
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
            
        print(f"Searching for top-{k} documents for {len(query_embeddings)} queries...")
        
        # Search FAISS index
        similarities, indices = self.index.search(query_embeddings, k)
        
        # Convert indices to doc_ids
        doc_ids = np.array([[self.evidence_doc_ids[idx] for idx in query_indices] 
                           for query_indices in indices])
        
        return similarities, doc_ids
    
    def batch_search_claims(self, claim_embeddings: np.ndarray, claim_ids: List[int], 
                           k: int = 50) -> dict:
        """Search for all claims and return structured results.
        
        Args:
            claim_embeddings: (n_claims, embed_dim) normalized claim embeddings
            claim_ids: List of claim_ids corresponding to embeddings
            k: Number of top documents to retrieve per claim
            
        Returns:
            Dictionary mapping claim_id -> list of (doc_id, similarity_score) tuples
        """
        similarities, doc_ids = self.search(claim_embeddings, k)
        
        results = {}
        for i, claim_id in enumerate(claim_ids):
            claim_results = []
            for j in range(k):
                doc_id = doc_ids[i, j]
                similarity = float(similarities[i, j])
                claim_results.append((doc_id, similarity))
            results[claim_id] = claim_results
            
        return results
