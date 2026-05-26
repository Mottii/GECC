import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "detectai-matplotlib"))

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from src.utils.visualization import save_or_show


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, show: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix - Gene Expression Cancer Classifier")
    plt.tight_layout()
    save_or_show(save_path, show=show)


def plot_roc_curves(y_true, y_proba, class_names, save_path=None, show: bool = False) -> None:
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(9, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - One-vs-Rest")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_or_show(save_path, show=show)


def plot_pca_embedding(X, y, class_names, save_path=None, show: bool = False) -> None:
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=0.65, s=22)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("PCA of Gene Expression Features")
    plt.legend()
    plt.tight_layout()
    save_or_show(save_path, show=show)


def full_evaluation(model, X_test, y_test, class_names, device, output_dir=None, show: bool = False):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        proba = model.predict_proba(X_t).cpu().numpy()
    y_pred = proba.argmax(axis=1)

    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(y_test, y_pred, class_names, output_dir / "confusion_matrix.png", show=show)
        plot_roc_curves(y_test, proba, class_names, output_dir / "roc_curves.png", show=show)
        plot_pca_embedding(X_test, y_test, class_names, output_dir / "pca_embedding.png", show=show)
    elif show:
        plot_confusion_matrix(y_test, y_pred, class_names, show=True)
        plot_roc_curves(y_test, proba, class_names, show=True)
        plot_pca_embedding(X_test, y_test, class_names, show=True)

    return y_pred, proba, report
