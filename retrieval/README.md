# Question 5: Information Retrieval System

FAISS-based IR system for scientific fact verification using SciFact dataset.

## Overview

Builds an information retrieval system to find relevant scientific papers for claims:
- **Documents**: 5,183 scientific paper abstracts (evidence)
- **Queries**: 809 scientific claims
- **Embeddings**: Pre-computed OpenAI embeddings
- **Search**: FAISS cosine similarity index

## Usage

### Run Complete Evaluation
```bash
python main.py
```

### Test Individual Components
```bash
python ground_truth.py    # Check ground truth extraction
python data_loader.py     # Test embedding loading
python faiss_retriever.py # Test FAISS search
```

## Results

- JSON output with IR performance metrics