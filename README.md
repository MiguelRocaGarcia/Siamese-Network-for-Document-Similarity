# Siamese Network for Document Similarity

This project implements and trains a **Siamese Neural Network** (using a triplet loss approach) to measure **document similarity** based on visual appearance and content. The goal is to automatically detect when two scanned documents are of the same template (form type) — a common task in document processing pipelines.

---

## Project Overview

The core idea is to embed each document into a fixed-length vector representation such that:

- **Similar documents (same template)** are mapped close together in embedding space.
- **Dissimilar documents (different templates)** are mapped far apart.

This is accomplished using a **ResNet50V2 backbone** (pretrained on ImageNet) as the feature extractor and a custom embedding head trained with **triplet loss**. The model was trained and validated on a **private dataset** of preprocessed scanned documents.  
> **Note:** The dataset used in this project is proprietary and not published.

---

## Process & Methodology

### 1. **Dataset**
- **≈ 9,675 documents** across multiple templates (one folder per template).
- Data split: **80% train / 20% test** at template level (so unseen templates appear only in test set).
- Generated **7,929 training triplets** (anchor, positive, negative) with:
  - `LIMIT_OF_TRIPLETS_PER_TEMPLATE = 20`
  - Additional semi-hard triplets included when available.

### 2. **Preprocessing**
- Border enhancement and normalization to improve OCR and visual consistency.
- Optional OCR-based augmentation and JSON metadata saving for downstream tasks.

### 3. **Model Architecture**
- **Backbone:** `ResNet50V2` (ImageNet weights) with first 3 blocks frozen.
- **Embedding size:** 2048.
- **Training setup:**
  - Loss: **Triplet Loss** with margin = 1.
  - Optimizer: Adam, learning rate `5e-5`.
  - Batch size: 128, epochs: 5.

### 4. **Training**
- The model was trained under a distributed `tf.distribute.Strategy` scope.
- After training, the model was exported (`.save()`) for later embedding generation.

### 5. **Evaluation**
- Embeddings were computed for all documents (train & test sets).
- Pairwise distances compared with a chosen threshold to produce **confusion matrices** and compute accuracy metrics.

---

## Results

### **Accuracy**
| Dataset        | Positive Accuracy | Negative Accuracy |
|---------------|-----------------|-----------------|
| **Train**     | **99.76%**      | **99.56%**      |
| **Test**      | **87.89%**      | **98.73%**      |

### **Confusion Matrices (Pairwise Evaluation)**

**Train Set:**
- **TP:** 39,250  **TN:** 4,626,517  
- **FP:** 20,528  **FN:** 96  
- **TPR:** 99.76%  **TNR:** 99.56%

**Test Set:**
- **TP:** 4,920  **TN:** 245,066  
- **FP:** 3,164  **FN:** 678  
- **TPR:** 87.89%  **TNR:** 98.73%

**Overall (All Embeddings):**
- **TP:** 44,170  **TN:** 7,034,822  
- **FP:** 43,659  **FN:** 774  
- **TPR:** 98.28%  **TNR:** 99.38%

---

## Testing & Clustering

The second notebook (`Testing Siamese Network v2.ipynb`) uses the trained model to:
- Generate embeddings for new document batches.
- Compute pairwise distances (example distance: `0.7041` between two sample docs).
- Cluster documents by template similarity and organize them into folders automatically.

---
