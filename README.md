
# Brain Tumor MRI Pipeline (Leakage-Free) — 28 Variants

This repo has my full workflow for a brain tumor MRI project:
1) **Clean the dataset** (remove duplicates + near-duplicates, prevent leakage)
2) **Train & evaluate a hybrid pipeline** (detection → classification) with **28 variants**
3) (Optional) evaluate on an **external TEST2** dataset (eval-only)

---

## What’s inside

### `remove_exact_and_near_duplicates.ipynb`
Cleans the dataset by removing:
- exact duplicates (SHA-256)
- pixel-level duplicates (resize → MD5)
- near-duplicates (pHash)

It also prevents **cross-class leakage**:  
if the same/near-same image appears in multiple classes, it drops that whole duplicate cluster.

 Output: a new clean folder like `*_CLEAN_NO_LEAK/` + reports in `_reports/`.



---

### `hybrid_brain_tumor_pipeline_leakage_free_28_variants.ipynb`
This is the **main notebook**.

What it does:
- Creates **leakage-free train/val/test split** using **MD5-grouping**
- Extracts backbone features and **caches** them (first run slow, re-run fast)
- Runs **28 variants** (different heads, FP/noFP, single/ensemble, stacking, OOF)
- Evaluates on:
  - **VAL**
  - **TEST**
  - **TEST2 (optional, eval-only)**

# Reproducible Brain Tumor MRI Classification, Localization, and Explainable AI Framework

## Overview

This repository provides the research code developed for brain tumor detection, classification, localization, segmentation, and explainable artificial intelligence analysis from magnetic resonance imaging (MRI).

The repository has been made publicly available to support transparency, reproducibility, independent verification, and future research. The code includes dataset-cleaning procedures, leakage-aware data splitting, model development, joint classification–segmentation training, external evaluation, and explainability analysis.

The framework classifies brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

This repository is intended for academic and research purposes only. It is not approved for direct clinical diagnosis or treatment decisions.

---

## Repository Contents


### `joint_classification_localization_framework.ipynb`

This notebook contains the joint multitask framework for brain tumor classification and tumor-region localization.

It includes:

- One shared multi-backbone encoder
- Joint classification and segmentation training
- A U-Net-style segmentation decoder
- Gated feature fusion
- A classification head using deep and handcrafted fingerprint features
- Weighted classification and segmentation losses
- Selection of one joint checkpoint using a combined validation score
- Classification evaluation using the four tumor categories
- Segmentation evaluation using Dice score
- Confusion matrices, ROC curves, training histories, and segmentation figures
- A Gradio-based inference interface

The implemented encoder uses the following five backbone architectures:

1. ConvNeXt-Base
2. EfficientNetV2-S
3. DenseNet-201
4. Swin Transformer-Small
5. Vision Transformer-Base

The classification and segmentation branches are optimized jointly. The total training objective is:

```text
Total loss = λcls × classification loss + λseg × segmentation loss


