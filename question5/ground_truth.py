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
    dataset = load_dataset("allenai/scifact", "claims")
    
    ground_truth = {}
    
    for item in dataset['train']:
        claim_id = item['id']
        evidence_doc_ids = set()
        
        # Extract evidence doc IDs from the evidence field
        if 'evidence' in item and item['evidence']:
            for evidence_group in item['evidence']:
                if isinstance(evidence_group, dict) and 'doc_id' in evidence_group:
                    evidence_doc_ids.add(evidence_group['doc_id'])
                elif isinstance(evidence_group, list):
                    for evidence_item in evidence_group:
                        if isinstance(evidence_item, dict) and 'doc_id' in evidence_item:
                            evidence_doc_ids.add(evidence_item['doc_id'])
        
        if evidence_doc_ids:
            ground_truth[claim_id] = evidence_doc_ids
    
    return ground_truth


if __name__ == "__main__":
    gt = load_scifact_ground_truth()
    print(f"Loaded ground truth for {len(gt)} claims")
    
    # Show sample
    sample_claim_id = list(gt.keys())[0]
    sample_evidence_ids = gt[sample_claim_id]
    print(f"Sample - Claim {sample_claim_id}: Evidence docs {sample_evidence_ids}")