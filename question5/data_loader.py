"""Data loading utilities for SciFact embeddings.

Author: Prabhav Singh
"""
import pickle
import numpy as np
from typing import Dict, List, Tuple


def load_embeddings() -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Load SciFact embeddings and create ID mappings.
    
    Returns:
        evidence_embeddings: (n_docs, embed_dim) numpy array
        claim_embeddings: (n_claims, embed_dim) numpy array  
        evidence_doc_ids: List of doc_ids corresponding to evidence_embeddings rows
        claim_ids: List of claim_ids corresponding to claim_embeddings rows
    """
    
    # Load evidence embeddings
    print("Loading evidence embeddings...")
    with open('data/scifact_evidence_embeddings.pkl', 'rb') as f:
        evidence_data = pickle.load(f)
    
    # Load claim embeddings  
    print("Loading claim embeddings...")
    with open('data/scifact_claim_embeddings.pkl', 'rb') as f:
        claim_data = pickle.load(f)
    
    # Extract evidence embeddings and doc_ids
    evidence_embeddings = []
    evidence_doc_ids = []
    
    for (doc_id, abstract_text), embedding in evidence_data.items():
        evidence_embeddings.append(embedding)
        evidence_doc_ids.append(doc_id)
    
    # Extract claim embeddings and claim_ids
    claim_embeddings = []
    claim_ids = []
    
    for (claim_id, claim_text), embedding in claim_data.items():
        claim_embeddings.append(embedding)
        claim_ids.append(claim_id)
    
    # Convert to numpy arrays
    evidence_embeddings = np.array(evidence_embeddings, dtype=np.float32)
    claim_embeddings = np.array(claim_embeddings, dtype=np.float32)
    
    # Normalize for cosine similarity
    evidence_embeddings = evidence_embeddings / np.linalg.norm(evidence_embeddings, axis=1, keepdims=True)
    claim_embeddings = claim_embeddings / np.linalg.norm(claim_embeddings, axis=1, keepdims=True)
    
    print(f"Loaded {len(evidence_embeddings)} evidence embeddings with dimension {evidence_embeddings.shape[1]}")
    print(f"Loaded {len(claim_embeddings)} claim embeddings with dimension {claim_embeddings.shape[1]}")
    
    return evidence_embeddings, claim_embeddings, evidence_doc_ids, claim_ids


if __name__ == "__main__":
    evidence_embs, claim_embs, evidence_ids, claim_ids = load_embeddings()
    print(f"Evidence shape: {evidence_embs.shape}")
    print(f"Claims shape: {claim_embs.shape}")
    print(f"Sample evidence ID: {evidence_ids[0]}")
    print(f"Sample claim ID: {claim_ids[0]}")