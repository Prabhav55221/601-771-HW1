"""Configuration for SciFact IR evaluation.

Author: Prabhav Singh
"""


class Config:
    """Configuration parameters for SciFact IR evaluation."""
    
    # Data paths
    EVIDENCE_EMBEDDINGS_PATH = "data/scifact_evidence_embeddings.pkl"
    CLAIM_EMBEDDINGS_PATH = "data/scifact_claim_embeddings.pkl"
    
    # FAISS configuration
    FAISS_INDEX_TYPE = "IndexFlatIP" 
    NORMALIZE_EMBEDDINGS = True
    
    # Retrieval configuration
    RETRIEVAL_K = 50
    
    # Evaluation configuration
    EVALUATION_K_VALUES = [1, 10, 50]
    
    # Output configuration
    RESULTS_DIR = "results"
    SAVE_DETAILED_RESULTS = True
    
    # SciFact dataset configuration
    SCIFACT_CLAIMS_SPLIT = "train"
    SCIFACT_CORPUS_SPLIT = "train"