"""Evaluation metrics for information retrieval.

Author: Prabhav Singh
"""
import numpy as np
from typing import Dict, Set, List, Tuple


def compute_mrr(retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
    """Compute Mean Reciprocal Rank at k.
    
    Args:
        retrieved_docs: List of retrieved document IDs (in rank order)
        relevant_docs: Set of relevant document IDs (ground truth)
        k: Cutoff for evaluation
        
    Returns:
        Reciprocal rank (1/rank of first relevant doc, 0 if none found in top-k)
    """
    if not relevant_docs:
        return 0.0
        
    for rank, doc_id in enumerate(retrieved_docs[:k], 1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    
    return 0.0


def compute_ap(retrieved_docs: List[int], relevant_docs: Set[int], k: int) -> float:
    """Compute Average Precision at k.
    
    Args:
        retrieved_docs: List of retrieved document IDs (in rank order)
        relevant_docs: Set of relevant document IDs (ground truth)
        k: Cutoff for evaluation
        
    Returns:
        Average precision score
    """
    if not relevant_docs:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for rank, doc_id in enumerate(retrieved_docs[:k], 1):
        if doc_id in relevant_docs:
            relevant_count += 1
            precision_at_rank = relevant_count / rank
            precision_sum += precision_at_rank
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / len(relevant_docs)


def evaluate_retrieval(retrieval_results: Dict[int, List[Tuple[int, float]]], 
                      ground_truth: Dict[int, Set[int]], 
                      k_values: List[int] = [1, 10, 50]) -> Dict[str, Dict[int, float]]:
    """Evaluate retrieval performance using MRR and MAP metrics.
    
    Args:
        retrieval_results: Dict mapping claim_id -> [(doc_id, similarity), ...]
        ground_truth: Dict mapping claim_id -> set of relevant doc_ids
        k_values: List of cutoff values to evaluate
        
    Returns:
        Dictionary with evaluation results:
        {
            'MRR': {1: mrr@1, 10: mrr@10, 50: mrr@50},
            'MAP': {1: map@1, 10: map@10, 50: map@50}
        }
    """
    results = {'MRR': {k: [] for k in k_values}, 'MAP': {k: [] for k in k_values}}
    
    evaluated_claims = 0
    
    for claim_id, retrieved_list in retrieval_results.items():
        # Skip claims without ground truth
        if claim_id not in ground_truth:
            continue
            
        relevant_docs = ground_truth[claim_id]
        retrieved_docs = [doc_id for doc_id, _ in retrieved_list]
        
        # Compute metrics for each k
        for k in k_values:
            mrr_score = compute_mrr(retrieved_docs, relevant_docs, k)
            map_score = compute_ap(retrieved_docs, relevant_docs, k)
            
            results['MRR'][k].append(mrr_score)
            results['MAP'][k].append(map_score)
        
        evaluated_claims += 1
    
    # Average across all claims
    final_results = {'MRR': {}, 'MAP': {}}
    for metric in ['MRR', 'MAP']:
        for k in k_values:
            if results[metric][k]:
                final_results[metric][k] = np.mean(results[metric][k])
            else:
                final_results[metric][k] = 0.0
    
    print(f"Evaluated {evaluated_claims} claims with ground truth")
    
    return final_results