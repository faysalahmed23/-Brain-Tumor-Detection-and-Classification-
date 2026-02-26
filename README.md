
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

 Outputs are saved in `OUT_DIR`.


