# Temporally Robust and Fair Credit Risk Prediction
**CSCI 567 Final Project — University of Southern California**

*Sadokat Khakimova, Hyeon Ah Jung, Shree Sriadibhatla, Zahra Iranmanesh*

---

## Overview

Credit risk models face two structural vulnerabilities in deployment: **temporal drift**, where year-over-year economic changes degrade predictive accuracy, and **demographic bias**, where performance optimization concentrates false-positive errors on disadvantaged groups.

We train six model families on 200,000 LendingClub loans (2007–2017) using a strict temporal split, searching hyperparameters jointly for temporal robustness and worst-case fairness via Optuna TPE. XGBoost achieves the best composite score on the 2016–2017 test set (Final Score 0.600, temporal AUC 0.683), with worst-case income-group FPR disparity of 0.060 — far below all competitors — while a post-processing threshold-equalization step reduces residual income disparities by a further 76–92% at negligible performance cost.

## Dataset

[LendingClub Loan Data — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

- 200,000 records sampled from public LendingClub loans (issue years 2007–2017)
- Binary target: loan default (Charged Off / Default)
- Temporal split: Train ≤2013 (n=20,286) | Val 2014–2015 (n=58,114) | Test 2016–2017 (n=77,616)

## How to Run

The main notebook is `credit_risk_temporal_fairness.ipynb`, designed to run on **Google Colab**.

1. Upload the notebook to [Google Colab](https://colab.research.google.com/)
2. Mount your Google Drive and place the LendingClub dataset (`accepted_2007_to_2018Q4.csv`) at the expected path in your Drive
3. Use **Runtime → Run all** to execute all cells in order

The notebook covers:
- Data loading, preprocessing, and temporal splitting
- Hyperparameter optimization with Optuna (60 trials per model family)
- Six model families: XGBoost, LightGBM, Random Forest, Neural Network (MLP), Logistic Regression, Decision Tree
- Fairness evaluation via nine-threshold worst-case sweep (FPR, TPR, PPR by income and credit group)
- Ablation studies (LR regularization, RF depth, XGB subsample)
- Post-processing per-group threshold equalization

## Key Results

| Model | Score | T-AUC | AUC Drop | TRP   | Inc FPR | Cr FPR |
|-------|-------|-------|----------|-------|---------|--------|
| XGB   | **0.600** | 0.683 | 0.016 | 0.677 | **0.060** | **0.022** |
| RF    | 0.336 | 0.680 | 0.016    | 0.675 | 0.177   | 0.078  |
| LGBM  | 0.185 | 0.681 | 0.033    | 0.676 | 0.327   | 0.147  |
| NN    | 0.014 | 0.676 | 0.016    | 0.668 | 0.184   | 0.070  |
| LR    | -0.153 | 0.677 | **0.012** | 0.669 | 0.313  | 0.163  |
| DTree | -0.342 | 0.670 | 0.018   | 0.669 | 0.287   | 0.164  |

*T-AUC: temporal-split AUC. AUC Drop: lower = more robust. Inc/Cr FPR: worst-case income/credit FPR disparity across 9-threshold sweep.*

## Libraries

Optuna, scikit-learn, XGBoost, LightGBM, pandas, numpy, matplotlib
