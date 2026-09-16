"""
Model Evaluation and Visualization Module for TRUTHSCAN AI.
Evaluates trained checkpoints on the untouched test dataset.
Generates genuine metric tables, confusion matrices, and comparison plots.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/batch environments
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple, Optional

import config
from src.dataset import load_processed_splits, NewsDataset
from src.tokenizer import Tokenizer
from models.textcnn_model import TextCNN
from models.lstm_model import LSTMClassifier


def load_model_from_checkpoint(model_type: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Loads a model instance with weights restored from a saved checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint["vocab_size"]

    if model_type == "cnn":
        cfg = checkpoint.get("config", config.CNN_CONFIG)
        model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=cfg["embedding_dim"],
            num_filters=cfg["num_filters"],
            kernel_sizes=cfg["kernel_sizes"],
            dropout=cfg["dropout"],
            padding_idx=config.PAD_IDX
        )
    elif model_type == "lstm":
        cfg = checkpoint.get("config", config.LSTM_CONFIG)
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            padding_idx=config.PAD_IDX
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_model_on_test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inference loop over the test dataset split.
    Returns (y_true, y_pred, y_probs).
    """
    model.eval()
    y_true_list = []
    y_probs_list = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            logits = model(sequences).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_probs_list.extend(probs)
            y_true_list.extend(labels.numpy())

    y_true = np.array(y_true_list)
    y_probs = np.array(y_probs_list)
    y_pred = (y_probs >= 0.5).astype(int)

    return y_true, y_pred, y_probs


def plot_loss_and_accuracy_curves():
    """Plots training vs validation loss and accuracy curves from history files."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    histories = {}
    for name, path in [("CNN", config.CNN_HISTORY_FILE), ("LSTM", config.LSTM_HISTORY_FILE)]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                histories[name] = json.load(f)

    if not histories:
        return

    # 1. Loss Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, h in histories.items():
        epochs = list(range(1, len(h["train_loss"]) + 1))
        axes[0].plot(epochs, h["train_loss"], label=f"{name} Train Loss", linestyle="--")
        axes[0].plot(epochs, h["val_loss"], label=f"{name} Val Loss", linewidth=2)
        
        axes[1].plot(epochs, [a * 100 for a in h["train_acc"]], label=f"{name} Train Acc", linestyle="--")
        axes[1].plot(epochs, [a * 100 for a in h["val_acc"]], label=f"{name} Val Acc", linewidth=2)

    axes[0].set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Binary Cross Entropy Loss", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].set_title("Training vs Validation Accuracy (%)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    loss_acc_path = os.path.join(config.PLOTS_DIR, "loss_accuracy_curves.png")
    plt.savefig(loss_acc_path, dpi=200)
    plt.close()


def plot_confusion_matrices(cm_dict: Dict[str, np.ndarray]):
    """Plots clean confusion matrix heatmaps for evaluated models."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    n = len(cm_dict)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    labels = ["Real (0)", "Fake (1)"]

    for ax, (name, cm) in zip(axes, cm_dict.items()):
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name} Confusion Matrix", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

        # Annotate text
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold")

    plt.tight_layout()
    cm_path = os.path.join(config.PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()


def plot_model_comparison_bar(df_comparison: pd.DataFrame):
    """Generates comparison bar chart between CNN and LSTM metrics."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    if df_comparison.empty:
        return

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, row in df_comparison.iterrows():
        model_name = row["Model"]
        values = [row[m] * 100 for m in metrics]
        offset = (idx - 0.5) * width if len(df_comparison) > 1 else 0
        bars = ax.bar(x + offset, values, width, label=model_name, alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("CNN vs LSTM Performance Comparison on Test Set", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    comp_plot_path = os.path.join(config.PLOTS_DIR, "model_comparison.png")
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()


def evaluate_models() -> Dict[str, Any]:
    """
    Evaluates whatever models have checkpoints available on the test dataset.
    Writes results/model_comparison.csv and generates comparison plots.
    """
    splits = load_processed_splits()
    if splits is None:
        raise FileNotFoundError("Processed test splits not found. Please train models first.")

    _, _, test_loader, tokenizer = splits
    device = config.DEVICE

    models_to_eval = []
    if os.path.exists(config.CNN_CHECKPOINT):
        models_to_eval.append(("CNN", config.CNN_CHECKPOINT, config.CNN_HISTORY_FILE))
    if os.path.exists(config.LSTM_CHECKPOINT):
        models_to_eval.append(("LSTM", config.LSTM_CHECKPOINT, config.LSTM_HISTORY_FILE))

    if not models_to_eval:
        raise FileNotFoundError(
            "Models not trained yet.\n"
            "Please train the models before viewing performance."
        )

    comparison_records = []
    confusion_matrices = {}
    detailed_reports = {}

    for model_name, ckpt_path, hist_path in models_to_eval:
        print(f"Evaluating {model_name} on untouched test set...")
        model = load_model_from_checkpoint(model_name.lower(), ckpt_path, device)
        y_true, y_pred, y_probs = evaluate_model_on_test(model, test_loader, device)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], output_dict=True)

        training_time = "N/A"
        if os.path.exists(hist_path):
            try:
                with open(hist_path, "r", encoding="utf-8") as f:
                    h = json.load(f)
                    sec = h.get("total_training_time", 0)
                    training_time = f"{sec:.1f}s"
            except Exception:
                pass

        comparison_records.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "Training Time": training_time
        })

        confusion_matrices[model_name] = cm
        detailed_reports[model_name] = report

    # Create and save comparison dataframe
    df_comparison = pd.DataFrame(comparison_records)
    df_comparison.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    print(f"\nSaved model comparison to: {config.MODEL_COMPARISON_CSV}")
    print(df_comparison.to_string(index=False))

    # Generate charts
    plot_loss_and_accuracy_curves()
    plot_confusion_matrices(confusion_matrices)
    plot_model_comparison_bar(df_comparison)

    return {
        "comparison_df": df_comparison,
        "confusion_matrices": confusion_matrices,
        "reports": detailed_reports
    }


if __name__ == "__main__":
    evaluate_models()
