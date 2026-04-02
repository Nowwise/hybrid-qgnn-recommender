# hybrid-qgnn-recommender
Hybrid Quantum–Classical Graph Neural Networks for Recommender Systems. Experiments comparing LightGCN and a Hybrid QGNN model, including full benchmark code, metrics, and reproducible pipelines.

## Student
**Danial K.T.M.**

## Supervisor
**Alexander**

## Degree Program
Master’s Thesis

---

## Project Description

This work is devoted to the study and development of hybrid quantum-enhanced graph neural network models for recommendation systems.  
The goal of the research is to explore whether the integration of quantum computing components into classical graph-based recommendation models can provide potential advantages in representation power, learning dynamics, or theoretical efficiency compared to fully classical approaches.

---

## Datasets (`dataset/`)

All benchmarks live under **`dataset/`** at the repository root — separate from `backend/`, `src/`, etc. Each benchmark is its **own subfolder** with the **same file layout**:

- `train.txt` — one line per user: `u i1 i2 i3 ...` (0-based ids, implicit positives)
- `test.txt` — same format for held-out interactions

| Path (`data_dir`) | Description |
|-------------------|-------------|
| **`dataset/amazon-book/`** | Primary benchmark (as in related work). Ship or obtain per your setup. |
| **`dataset/movielens-100k/`** | MovieLens-100K, implicit feedback (ratings ≥ 4), **per-user temporal leave-last-out** (last interaction → test). Generated locally — not committed if you prefer to run the script yourself. 

**Create MovieLens-100K files** (writes `dataset/movielens-100k/`, downloads [GroupLens ml-100k.zip](https://grouplens.org/datasets/movielens/100k/)):

Set `data_dir` to `dataset/movielens-100k` in the dashboard or API to train on that split. Default is `dataset/amazon-book`.

**Docker:** `docker-compose` mounts `./dataset` → `/app/dataset` so the API sees the same paths.

---

## Work Plan (by Checkpoints)

### 1. Literature Review and Problem Formulation  
**Timeline:** December - January

- Review classical recommendation systems and collaborative filtering approaches  
- Study graph-based recommender systems (GCN, LightGCN, message-passing GNNs)  
- Analyze existing research on quantum machine learning and quantum graph models  
- Review hybrid quantum–classical neural network architectures  
- Identify limitations of classical approaches and formulate research questions  

**Output:** literature review, problem statement, research objectives

---

### 2. Data Selection and Preparation  
**Timeline:** January

- Selection of benchmark datasets for recommendation systems  
- Construction of user–item interaction graphs  
- Data preprocessing and filtering  
- Exploratory data analysis (EDA) of graph structure and sparsity  

**Output:** prepared datasets, EDA results

---

### 3. Baseline Classical Models  
**Timeline:** Late January – February

- Implementation of baseline recommendation models  
- Training of classical graph-based models (e.g. LightGCN)  
- Definition and calculation of evaluation metrics (AUC, Recall@K, NDCG@K)  
- Analysis of baseline performance  

**Output:** baseline models and evaluation results

---

### 4. Hybrid Quantum-Enhanced Model Design  
**Timeline:** February – March

- Design of a hybrid classical–quantum model architecture  
- Definition of quantum components within the GNN pipeline  
- Selection of quantum encoding and circuit structure  
- Theoretical justification of the proposed hybrid approach  

**Output:** model architecture and formal description

---

### 5. Implementation of Hybrid QGNN Model  
**Timeline:** March – April

- Implementation of the hybrid model using classical ML frameworks  
- Integration of quantum components via quantum simulators  
- Training and inference pipeline development  
- Experiment reproducibility setup  

**Output:** working hybrid model implementation

---

### 6. Experimental Evaluation and Comparison  
**Timeline:** April

- Comparative evaluation of classical and hybrid models  
- Performance analysis across selected metrics  
- Ablation studies of quantum components  
- Discussion of scalability and simulation limitations  

**Output:** experimental results and analysis

---

### 7. Results Analysis and Discussion  
**Timeline:** Late April – May

- Interpretation of experimental findings  
- Analysis of strengths and limitations of the hybrid approach  
- Discussion of quantum noise, simulators, and practical feasibility  
- Positioning results within current research landscape  

**Output:** analytical discussion section

---

### 8. Thesis Writing and Finalization  
**Timeline:** May – June

- Writing all thesis chapters  
- Preparation of figures, tables, and references  
- Final editing and formatting  

**Output:** completed Master’s thesis (ВКР)

---

## Evaluation protocol (implemented)

- **Model selection during training:** best checkpoints are chosen by **validation AUC** on the stratified user–item pair split (`lg_best.pt`, `hyb_best.pt`; rows in `metrics.csv` with `split=val`).
- **Ranking metrics:** after training, the pipeline evaluates sampled **Recall@K** and **NDCG@K** (default K in `ranking_ks`, e.g. 5, 10, 20). Each query uses **one positive** from the evaluation split and **`ranking_negatives`** random negatives that are not in that user’s **training** positives. This is a standard **implicit-feedback** sanity check; it is **not** full-catalog ranking (which would be much more expensive).
- **Splits:** `val_ranking` uses validation positives from the same split as training val. **`test_ranking`** (if `eval_test_ranking` is true) uses interactions from **`test.txt`**, restricted to users who appear in the training pair matrix — so it is a **held-out interaction** check, not cold-start users.
- **Ablation:** rows for **`HybridQGNN (ablation classical head)`** run the same trained hybrid with **`force_classical`** (encoder → linear `fallback` → MLP head, **no** quantum block forward), to separate the effect of the quantum feature map from the rest of the architecture.
- **Cost / feasibility:** **`phase_timings.json`** stores wall-clock seconds per phase (`prepare_data`, each epoch, `ranking_evaluation`, `analysis_export`). The same timings are duplicated in `metrics.csv` with `split=timing` for plotting.
- **Tables:** `write_comparative_tables` writes `val_best_comparative.csv`, `val_metrics_per_epoch.csv`, **`ranking_comparative.csv`** (when ranking rows exist), and **`full_model_comparative.csv`** — one wide table with val@best metrics, `val_rank_*`, `test_rank_*`, and **Δ (Hybrid − LightGCN)**. The dashboard loads this (or rebuilds it from `metrics.csv` if the file is missing).

### Reproducibility and variance

- Each run saves **`run_config.json`** including **`seed`**. For thesis reporting of variance, re-run with different seeds (e.g. 41–45) and report mean ± std; the dashboard/API accept `seed` overrides.

---

## Notes

- The plan may be refined and updated during subsequent checkpoints  
- Timeline and scope may be adjusted based on experimental results and supervisor feedback
