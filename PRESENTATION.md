# Presentation Outline — DetectAI

## Slide 1: Title
- DetectAI: Gene Expression Cancer Classifier
- [Your Name]
- Track: Deep Learning

## Slide 2: Problem
- Problem: Identifying cancer type from RNA-Seq data is time-consuming
- User: Bioinformaticians and clinical researchers
- Solution: ML classifier + interactive API

## Slide 3: Data
- Source: UCI TCGA-PANCAN-HiSeq (801 samples, 20531 genes)
- 5 cancer classes: BRCA, KIRC, COAD, LUAD, PRAD
- Preprocessing: top-2000 variance features, StandardScaler

## Slide 4: Approach
- Baseline: LR → RF → SVM → XGBoost (cross-validated)
- Main: Deep Neural Network (4 hidden layers, BatchNorm, Dropout)
- Why DNN? Captures non-linear gene interactions

## Slide 5: Results
- DNN: 99.38% test accuracy vs ~97-98% baselines
- Confusion matrix, ROC curves, PCA visualization
- API + frontend for interactive use

## Slide 6: Error Analysis
- 1-2 misclassifications (LUAD↔BRCA confusion)
- Low-confidence flagging (<0.75)
- All errors documented with sample-level analysis

## Slide 7: Limitations
- Small dataset (801 samples)
- Simple variance-based feature selection
- No external validation
- Softmax not calibrated

## Slide 8: What I Changed vs Standard Approach
- Added PCA visualization for interpretability
- Confidence thresholding with user-facing warnings
- BatchNorm + aggressive Dropout (0.4) for regularization
- Clinical assistant chat interface
- Full Docker + API deployment

## Slide 9: Conclusion & Future Work
- Strong result (99.38%) with robust evaluation
- Future: external validation, calibration, biologically-informed features
- Code: https://github.com/Mottii/GECC
