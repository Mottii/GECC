# Final Report — DetectAI: Gene Expression Cancer Classifier

## 1. Introduction
Cancer type identification from gene expression data is a key bioinformatics task. This project builds a deep neural network classifier that predicts one of five cancer types from RNA-Seq expression profiles, with a REST API and interactive web frontend for researchers.

## 2. Data
- **Source:** UCI TCGA-PANCAN-HiSeq (801 samples, 20531 genes)
- **Classes:** BRCA (300), KIRC (146), COAD (78), LUAD (154), PRAD (123) — roughly balanced
- **Preprocessing:** Top-2000 variance-based feature selection → StandardScaler → 80/20 stratified split (640 train, 161 test)

## 3. Methods

### Baseline Models
Four classic ML classifiers: Logistic Regression, Random Forest (300 estimators), SVM (RBF kernel), XGBoost (250 estimators). Evaluated via stratified 5-fold cross-validation.

### Main Model: Deep Neural Network
Architecture: Input(2000) → Linear(1024)→BN→ReLU→Dropout(0.4) → Linear(512)→BN→ReLU→Dropout(0.4) → Linear(256)→BN→ReLU→Dropout(0.4) → Linear(128)→BN→ReLU→Dropout(0.4) → Linear(5)

- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Early stopping: patience=10
- Gradient clipping: max_norm=1.0
- Batch size: 32, Epochs: 100 (stopped at 11)

## 4. Results

| Model | Test Accuracy | F1-macro |
|-------|-------------|----------|
| Logistic Regression | ~96% | ~0.96 |
| Random Forest | ~97% | ~0.97 |
| SVM | ~97% | ~0.97 |
| XGBoost | ~98% | ~0.98 |
| **DNN (ours)** | **99.38%** | **~0.99** |

- Confusion matrix shows only 1-2 misclassifications across 161 test samples
- ROC-AUC > 0.99 for all classes (one-vs-rest)
- PCA embedding plot shows clear class separation

## 5. Error Analysis
- Majority of misclassifications occur between LUAD (lung) and BRCA (breast) — potentially due to shared epithelial expression signatures
- Low-confidence predictions (<0.75 confidence) are flagged in the API
- The small dataset (801 samples) raises generalizability concerns beyond TCGA cohorts

## 6. Limitations
- **Small cohort:** 801 samples from TCGA may not generalize to independent patient populations
- **Feature selection:** Variance-based top-K is biologically naive; ignores low-variance but important regulatory genes
- **No calibration:** Softmax probabilities are not calibrated — reliability diagrams were not computed
- **Overfitting risk:** 100% validation accuracy at epoch 11 suggests possible overfitting despite Dropout
- **Gemini API optional:** Clinical assistant requires external API key for full functionality

## 7. Reproducibility
- Global seeding (torch + numpy + random + cudnn)
- Training manifest with run ID, feature hash, environment snapshot
- Docker support with docker-compose
- Automated data download script

## 8. Conclusion
The DNN classifier achieves excellent accuracy (99.38%) on the TCGA-PANCAN dataset, outperforming all baseline models. The interactive frontend and API make it a practical tool for demonstration and exploration. Future work should focus on external validation, model calibration, and biologically informed feature selection.
