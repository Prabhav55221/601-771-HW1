"""Main experiment runner for SciFact IR evaluation.

Author: Prabhav Singh
"""
import json
import os
from ground_truth import load_scifact_ground_truth
from data_loader import load_embeddings
from faiss_retriever import FAISSRetriever
from evaluator import evaluate_retrieval


def main():
    """Run complete IR evaluation experiment."""
    print("Starting SciFact Information Retrieval Evaluation")
    print("=" * 60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # 1. Load ground truth
    print("\n1. Loading ground truth...")
    ground_truth = load_scifact_ground_truth()
    print(f"Loaded ground truth for {len(ground_truth)} claims")
    
    # 2. Load embeddings
    print("\n2. Loading embeddings...")
    evidence_embeddings, claim_embeddings, evidence_doc_ids, claim_ids = load_embeddings()
    
    # 3. Build FAISS index
    print("\n3. Building FAISS retrieval index...")
    retriever = FAISSRetriever()
    retriever.build_index(evidence_embeddings, evidence_doc_ids)
    
    # 4. Perform retrieval
    print("\n4. Performing retrieval for all claims...")
    retrieval_results = retriever.batch_search_claims(claim_embeddings, claim_ids, k=50)
    print(f"Retrieved results for {len(retrieval_results)} claims")
    
    # 5. Evaluate performance
    print("\n5. Evaluating retrieval performance...")
    evaluation_results = evaluate_retrieval(retrieval_results, ground_truth, k_values=[1, 10, 50])
    
    # 6. Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nOpenAI Embeddings with FAISS (Cosine Similarity):")
    print("-" * 50)
    
    for metric in ['MRR', 'MAP']:
        print(f"\n{metric} Results:")
        for k in [1, 10, 50]:
            score = evaluation_results[metric][k]
            print(f"  {metric}@{k}: {score:.4f}")
    
    # 7. Save detailed results
    print("\n6. Saving results...")
    
    # Prepare results for table format
    table_results = {
        "OpenAI_Embeddings": {
            "MRR@1": evaluation_results['MRR'][1],
            "MAP@1": evaluation_results['MAP'][1], 
            "MRR@10": evaluation_results['MRR'][10],
            "MAP@10": evaluation_results['MAP'][10],
            "MRR@50": evaluation_results['MRR'][50],
            "MAP@50": evaluation_results['MAP'][50]
        },
        "ElasticSearch": {
            "MRR@1": None,
            "MAP@1": None,
            "MRR@10": None,
            "MAP@10": None, 
            "MRR@50": None,
            "MAP@50": None
        }
    }
    
    # Save as JSON
    with open('results/ir_evaluation_results.json', 'w') as f:
        json.dump(table_results, f, indent=2)
    
    # Save detailed results
    detailed_results = {
        "experiment_info": {
            "total_claims": len(claim_ids),
            "total_evidence_docs": len(evidence_doc_ids),
            "claims_with_ground_truth": len(ground_truth),
            "retrieval_cutoff": 50
        },
        "evaluation_results": evaluation_results,
        "table_format": table_results
    }
    
    with open('results/detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("Results saved to:")
    print("  - results/ir_evaluation_results.json (table format)")
    print("  - results/detailed_results.json (full details)")
    
    # Print table format for easy copying
    print("\n" + "=" * 60)
    print("TABLE FORMAT (for LaTeX)")
    print("=" * 60)
    
    openai_results = table_results["OpenAI_Embeddings"]
    print("OpenAI Embeddings &", 
          f"{openai_results['MRR@1']:.4f} &",
          f"{openai_results['MAP@1']:.4f} &", 
          f"{openai_results['MRR@10']:.4f} &",
          f"{openai_results['MAP@10']:.4f} &",
          f"{openai_results['MRR@50']:.4f} &",
          f"{openai_results['MAP@50']:.4f} \\\\")
    print("ElasticSearch & & & & & & \\\\")
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()