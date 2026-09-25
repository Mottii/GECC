# Work Breakdown Structure — DetectAI

## Week 1: Setup & Data
- [x] Define problem and select track
- [x] Find and download TCGA-PANCAN dataset
- [x] Set up project structure (src/, app/, tests/)
- [x] Implement data download & preprocessing pipeline

## Week 2: Baseline & EDA
- [x] Implement EDA (distributions, class balance, PCA visualization)
- [x] Train baseline models (LR, RF, SVM, XGBoost)
- [x] Build cross-validation pipeline
- [x] Document baseline performance

## Week 3: Main Solution
- [x] Implement DNN architecture (4 hidden layers, BatchNorm, Dropout)
- [x] Implement training loop (AdamW, CosineAnnealingLR, early stopping)
- [x] Hyperparameter tuning (random search)
- [x] Evaluate final model (classification report, confusion matrix, ROC curves)

## Week 4: API, Frontend & Documentation
- [x] Build FastAPI REST API (predict, health, metadata endpoints)
- [x] Build interactive frontend (PCA scatter, chat assistant, file upload)
- [x] Write tests (API, inference, model, preprocessing)
- [x] Dockerize application
- [x] Write documentation and create archive
- [ ] Submit deliverables
