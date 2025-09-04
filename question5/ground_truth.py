"""Ground truth extraction for SciFact IR evaluation.

Author: Prabhav Singh
"""
from datasets import load_dataset
from typing import Dict, Set


def load_scifact_ground_truth() -> Dict[int, Set[int]]:
    """Load SciFact ground truth mappings from HuggingFace dataset.
    
    Returns:
        Dictionary mapping claim_id -> set of relevant evidence_doc_ids
    """
    dataset = load_dataset("allenai/scifact", "claims", trust_remote_code=True)
    
    ground_truth = {}
    
    for item in dataset['train']:
        claim_id = item['id']
        evidence_doc_ids = set()
        
        # Use cited_doc_ids as ground truth (these are the relevant documents)
        if 'cited_doc_ids' in item and item['cited_doc_ids']:
            for doc_id in item['cited_doc_ids']:
                evidence_doc_ids.add(doc_id)
        
        # Also include evidence_doc_id if present (specific evidence document)
        if 'evidence_doc_id' in item and item['evidence_doc_id']:
            try:
                evidence_doc_id = int(item['evidence_doc_id'])
                evidence_doc_ids.add(evidence_doc_id)
            except (ValueError, TypeError):
                pass  # Skip if not a valid integer
        
        if evidence_doc_ids:
            ground_truth[claim_id] = evidence_doc_ids
    
    return ground_truth


if __name__ == "__main__":
    gt = load_scifact_ground_truth()
    print(f"Loaded ground truth for {len(gt)} claims")
    
    if gt:
        # Show samples
        sample_items = list(gt.items())[:3]
        for claim_id, evidence_ids in sample_items:
            print(f"Claim {claim_id}: Evidence docs {evidence_ids}")
    else:
        print("No ground truth loaded - check dataset structure")