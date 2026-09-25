# Action Plan — DetectAI: Gene Expression Cancer Classifier

**Project Title:** DetectAI — Gene Expression Cancer Classification
**Track:** Deep Learning (Track 2) / Classic ML (Track 1)
**Author:** [Your Name]

## Problem Statement
Classify cancer type from RNA-Seq gene expression profiles to assist biomedical researchers in快速 identifying tumor origins from transcriptomic data.

## Target User
Bioinformaticians and clinical researchers who need an interactive tool to classify cancer samples and explore gene expression patterns.

## Data
- **Source:** UCI TCGA-PANCAN-HiSeq dataset (801 samples, 20531 genes)
- **Classes:** 5 cancer types (BRCA, KIRC, COAD, LUAD, PRAD)
- **Split:** 80/20 stratified train/test

## Model Input
Vector of 2000 normalized gene expression values (top-K variance selected, standardized).

## Model Output
Predicted cancer class with confidence score, per-class probabilities, and PCA visualization coordinates.

## Baseline
- Logistic Regression (max_iter=2000)
- Random Forest (300 estimators)
- SVM (RBF kernel)
- XGBoost (250 estimators)

## Main Solution
Deep Neural Network (4 hidden layers: 1024→512→256→128) with BatchNorm, Dropout(0.4), AdamW optimizer, CosineAnnealingLR scheduler, and early stopping.

## Evaluation Metrics
- Primary: Accuracy, F1-macro
- Secondary: Per-class precision/recall/F1, confusion matrix, ROC-AUC curves
- Confidence calibration check

## Success Criteria
- Beat baseline models on test accuracy and F1-macro
- Working REST API with interactive frontend
- Reproducible pipeline with seeded randomness
- Error analysis and documented limitations
