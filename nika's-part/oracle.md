# FrugalProver Pilot - Computational Results

## Dataset
- Source: GSM8K
- Samples: 500
- Features: 768 (DeBERTa embeddings)
- Train/Test split: 80/20%

---

## Architecture Performance

| Architecture | R2 | MAE | Accuracy | CV R2 | Train Time (s) |
|-------------|----|-----|----------|-------|----------------|
| MLP (medium) | 0.803 | 0.585 | 0.600 | 0.434 | 1.20 |
| MLP (small) | 0.794 | 0.601 | 0.480 | 0.368 | 0.55 |
| Linear (Lasso) | 0.586 | 0.729 | 0.530 | 0.375 | 0.04 |
| Gradient Boosting | 0.539 | 0.700 | 0.580 | 0.098 | 15.52 |
| Random Forest (300) | 0.482 | 0.728 | 0.610 | 0.431 | 46.44 |
| Random Forest (100) | 0.436 | 0.766 | 0.580 | 0.432 | 16.81 |
| Linear (Ridge) | 0.371 | 1.093 | 0.270 | 0.275 | 0.26 |

---

## Best Model: MLP (medium)

| Parameter | Value |
|-----------|-------|
| Architecture | 256-128-64 |
| R2 | 0.8032 |
| MAE | 0.5851 |
| Accuracy | 0.6000 |
| CV R2 | 0.4337 |
| CV R2 Std | 0.0469 |
| Train Time | 1.20s |

---

## Correlation Analysis

| Metric | Value | p-value |
|--------|-------|---------|
| Spearman (mean activations) | -0.239 | 0.017 |
| Max neuron correlation | 0.704 | - |
| Mean correlation (100 neurons) | 0.418 | - |

---

## Baseline Comparison

| Model | R2 | Improvement |
|-------|----|-------------|
| MLP (medium) | 0.803 | +0.803 |
| Baseline (mean) | 0.000 | - |

| Metric | Baseline | MLP (medium) | Improvement |
|--------|----------|--------------|-------------|
| MAE | 2.500 | 0.585 | 4.3x |

---

## Budget Distribution

| Budget Range | Samples | Percent |
|--------------|---------|---------|
| 1-3 | 140 | 28% |
| 5-8 | 180 | 36% |
| 13-21 | 180 | 36% |

---

## Hardware

| Component | Specification |
|-----------|---------------|
| Platform | Google Colab |
| GPU | T4 (if available) |
| CPU | Intel Xeon |
| RAM | 12GB |

---

## Execution Time

| Phase | Time |
|-------|------|
| Data Loading | ~5s |
| Model Loading | ~10s |
| Activation Extraction | ~60s |
| Model Training (all) | ~80s |
| **Total** | **~155s** |

---

## Results Files

| File | Description |
|------|-------------|
| gsm8k_results.json | Raw metrics |
| gsm8k_activations.npy | Extracted activations (500x768) |
| gsm8k_budgets.npy | Budget labels (500x1) |
| gsm8k_results.png | Visualization plots |

---

## Summary

| Metric | Value |
|--------|-------|
| Best R2 | 0.803 |
| Best MAE | 0.585 |
| Best Model | MLP (medium) |
| Hypothesis H1 | CONFIRMED |
