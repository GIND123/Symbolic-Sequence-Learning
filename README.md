## Using Next-Gen Transformers to Seed Generative Models for Symbolic Regression

**Author:** Govind  
**Organization:** ML4SCI (Machine Learning for Science)  
**Mentors:** Eric Reinhardt, Dinesh Ramakrishnan, Sergei Gleyzer, Nobuchika Okada, Ritesh Bhalerao

---

## Overview

The core idea: use a **Set Transformer** to predict candidate symbolic expressions from numerical data, then feed those predictions as the **initial population for Genetic Programming**, which refines them via evolution.

### Results Summary

| Method | Mean R² | Median R² | R² > 0.99 | NMSE < 0.01 |
|---|---|---|---|---|
| Transformer Only | 0.106 | 0.152 | 4/10 | 2/10 |
| GP Only (Random Init) | 0.896 | 0.954 | 3/10 | 1/10 |
| **Transformer + GP (Ours)** | **0.930** | **0.999** | **8/10** | **4/10** |

**Ablation:** Transformer seeding improved NMSE by **52.8%** on average over random GP initialization, and helped on **10/10 test equations**.

---

## Architecture

```
  INPUT: {(x₁,...,xₖ, y)} × N points
              │
  ┌───────────▼───────────┐
  │  SET TRANSFORMER ENC  │  ISAB layers (permutation-invariant)
  │  → PMA (16 latent)    │  O(n·m) complexity
  └───────────┬───────────┘
              │ cross-attention
  ┌───────────▼───────────┐
  │  AUTOREGRESSIVE DEC   │  Causal Transformer (4 layers)
  │  → prefix tokens      │  Teacher-forced training
  └───────────┬───────────┘
              │ beam search (K=15)
  ┌───────────▼───────────┐
  │  TOP-K CANDIDATES     │  Prefix → expression trees
  └───────────┬───────────┘
              │ seed population
  ┌───────────▼───────────┐
  │  GENETIC PROGRAMMING  │  Crossover + mutation + elitism
  │  + constant optim     │  75 generations
  └───────────┬───────────┘
              │
  OUTPUT: Symbolic Expression
```

---

## Task 1.1 — Tokenization

**Strategy: Prefix (Polish) Notation**

| Property | Why it matters |
|---|---|
| No parentheses needed | Smaller vocabulary (60 tokens), shorter sequences (mean 14.7) |
| 1-to-1 tree mapping | O(n) conversion to GP individuals via recursive parser |
| Fixed arity | Enables constrained decoding during beam search |
| Literature standard | SymbolicGPT, NESYMRES, E2E-SR all use prefix notation |
| Left-folded binary ops | Consistent binary trees for GP subtree crossover |

**Example:**
```
Formula:  G*m1*m2/((x2-x1)² + (y2-y1)²)
Variables: m1→x1, m2→x2, G→x3, x1→x4, x2→x5, y1→x6, y2→x7
Prefix:   [mul, mul, mul, x3, x4, x5, div, 1, add, pow, add, x5, neg, x4, 2, pow, add, x7, neg, x6, 2]
```

---

## Task 2.6 — Model & Evaluation

### Training
- **Data augmentation**: Each equation sampled 50× per epoch with different random observation points + Gaussian noise → 3,850 effective training samples (prevents overfitting on 77 equations)
- **Label smoothing** (0.1) + **weight decay** (1e-2) for regularization
- **Warmup + cosine LR schedule**
- **Mixed precision** (AMP) for GPU efficiency

### Evaluation Pipeline
For each test equation:
1. **Transformer beam search** → 15 candidate expressions
2. **Transformer+GP**: Seed GP population with beam candidates + mutated variants
3. **GP-only baseline**: Random initial population (same hyperparameters)
4. Compare NMSE and R² across all three methods


<img width="1590" height="490" alt="image" src="https://github.com/user-attachments/assets/94480e0e-1de3-45bd-bcc2-ff51de0e0822" />

---

## Project Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── tokenizer.py          # Task 1.1: Prefix notation tokenization
│   ├── dataset.py             # Feynman dataset + augmentation
│   ├── model.py               # Set Transformer encoder + decoder
│   ├── genetic_programming.py # GP engine with transformer seeding
│   └── train.py               # Training + evaluation pipeline
├── notebooks/
│   └── notebook.py       # Single-file Colab notebook (run top-to-bottom)
└── results/
    └── results.json           # Evaluation metrics
```

---

## Quick Start

### Google Colab (recommended)
1. Upload `notebooks/gsoc_notebook.py` to Colab
2. Enable GPU runtime
3. Update the two path variables at the top:
   ```python
   EQUATIONS_CSV = "/content/drive/MyDrive/Papers/GSOC/FeynmanEquations.csv"
   DATA_DIR = "/content/Feynman_with_units"
   ```
4. Run all cells

### Local
```bash
pip install -r requirements.txt

# Download data
wget https://space.mit.edu/home/tegmark/aifeynman/Feynman_with_units.tar.gz
tar -xzf Feynman_with_units.tar.gz
# Download FeynmanEquations.csv from the same page

# Run
cd src
python train.py --csv FeynmanEquations.csv --data_dir Feynman_with_units
```

---

## Proposed Improvements

1. **Synthetic data pretraining**: Train on 100K+ randomly generated expressions before fine-tuning on Feynman — addresses the core data scarcity bottleneck
2. **Constrained beam search**: Mask invalid tokens at each decoding step using prefix arity rules to guarantee syntactically valid candidates
3. **Constant optimization**: After GP finds tree structure, use L-BFGS-B to optimize numeric leaf constants
4. **Modern transformer architectures**: Explore Titans, Mamba, or other efficient attention mechanisms for the encoder
5. **RL-based refinement**: Use expression fitness as reward signal to fine-tune the decoder via REINFORCE

---

## References

- [SymbolicGPT](https://arxiv.org/abs/2106.14131) — Li et al., 2022
- [Set Transformer](https://arxiv.org/abs/1810.00825) — Lee et al., 2019
- [AI Feynman](https://arxiv.org/abs/1905.11481) — Udrescu & Tegmark, 2019
- [E2E Transformers for SR](https://arxiv.org/abs/2106.06427) — Kamienny et al., 2022
- [NESYMRES](https://arxiv.org/abs/2106.06427) — Biggio et al., 2021
