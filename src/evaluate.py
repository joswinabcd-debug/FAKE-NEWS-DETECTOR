"""
Model Evaluation and Visualization Module for TRUTHSCAN AI.
Evaluates trained checkpoints on the untouched test dataset.
Generates genuine metric tables, confusion matrices, and comparison plots
styled with Dark AI theme aesthetics.
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


# Dark AI Theme Styling Configuration for Matplotlib
DARK_BG = "#0B120E"
DARKER_BG = "#050807"
ACCENT_GREEN = "#00E676"
ACCENT_GREEN_MUTED = "#00B85A"
TEXT_COLOR = "#FFFFFF"
MUTED_TEXT = "#A7B5AD"
GRID_COLOR = "#152B1E"
CARD_BORDER = "#1B3828"


def apply_dark_theme(fig, axes):
    """Applies Dark AI Dashboard styling to matplotlib figures."""
    fig.patch.set_facecolor(DARKER_BG)
    axes_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in axes_list:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=MUTED_TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(CARD_BORDER)


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

    state_dict = checkpoint["model_state_dict"]
    
    # Gracefully handle loading older 1-stage TextCNN weights into updated model
    if model_type == "cnn" and "fc.weight" in state_dict and "fc1.weight" not in state_dict:
        model.fc1 = nn.Identity()
        model.fc2 = nn.Linear(cfg["num_filters"] * len(cfg["kernel_sizes"]), 1)
        model.fc2.weight.data = state_dict["fc.weight"]
        model.fc2.bias.data = state_dict["fc.bias"]
        filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        model.load_state_dict(filtered_state, strict=False)
    else:
        model.load_state_dict(state_dict)

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

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    apply_dark_theme(fig, axes)
    
    palette = {
        "CNN": {"train": "#38BDF8", "val": "#00E676"},
        "LSTM": {"train": "#F59E0B", "val": "#EC4899"}
    }

    for name, h in histories.items():
        epochs = list(range(1, len(h["train_loss"]) + 1))
        c_train = palette.get(name, {}).get("train", "#94A3B8")
        c_val = palette.get(name, {}).get("val", ACCENT_GREEN)
        
        axes[0].plot(epochs, h["train_loss"], label=f"{name} Train Loss", linestyle="--", color=c_train, alpha=0.8)
        axes[0].plot(epochs, h["val_loss"], label=f"{name} Val Loss", linewidth=2.2, color=c_val)
        
        axes[1].plot(epochs, [a * 100 for a in h["train_acc"]], label=f"{name} Train Acc", linestyle="--", color=c_train, alpha=0.8)
        axes[1].plot(epochs, [a * 100 for a in h["val_acc"]], label=f"{name} Val Acc", linewidth=2.2, color=c_val)

    axes[0].set_title("Training vs Validation Loss", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Binary Cross Entropy Loss", fontsize=10)
    axes[0].legend(facecolor=DARK_BG, edgecolor=CARD_BORDER, labelcolor=TEXT_COLOR, fontsize=8.5)
    axes[0].grid(True, linestyle=":", color=GRID_COLOR, alpha=0.7)

    axes[1].set_title("Training vs Validation Accuracy (%)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy (%)", fontsize=10)
    axes[1].legend(facecolor=DARK_BG, edgecolor=CARD_BORDER, labelcolor=TEXT_COLOR, fontsize=8.5)
    axes[1].grid(True, linestyle=":", color=GRID_COLOR, alpha=0.7)

    plt.tight_layout()
    loss_acc_path = os.path.join(config.PLOTS_DIR, "loss_accuracy_curves.png")
    plt.savefig(loss_acc_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()


def plot_confusion_matrices(cm_dict: Dict[str, np.ndarray]):
    """Plots clean confusion matrix heatmaps for evaluated models."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    n = len(cm_dict)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.6))
    if n == 1:
        axes = [axes]

    apply_dark_theme(fig, axes)
    labels = ["Real (0)", "Fake (1)"]

    for ax, (name, cm) in zip(axes, cm_dict.items()):
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        ax.set_title(f"{name} Confusion Matrix", fontsize=11, fontweight="bold", pad=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=MUTED_TEXT)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=MUTED_TEXT)
        
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="black" if cm[i, j] > thresh else "white",
                        fontsize=11, fontweight="bold")

    plt.tight_layout()
    cm_path = os.path.join(config.PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()


def plot_model_comparison_bar(df_comparison: pd.DataFrame):
    """Generates comparison bar chart between CNN and LSTM metrics."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    if df_comparison.empty:
        return

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    apply_dark_theme(fig, ax)

    colors = ["#00E676", "#38BDF8"]

    for idx, row in df_comparison.iterrows():
        model_name = row["Model"]
        values = [row[m] * 100 for m in metrics]
        offset = (idx - 0.5) * width if len(df_comparison) > 1 else 0
        c = colors[idx % len(colors)]
        bars = ax.bar(x + offset, values, width, label=model_name, color=c, alpha=0.9, edgecolor=CARD_BORDER)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=TEXT_COLOR)

    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_title("CNN vs LSTM Performance Comparison on Test Set", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(facecolor=DARK_BG, edgecolor=CARD_BORDER, labelcolor=TEXT_COLOR, loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", color=GRID_COLOR, alpha=0.7)

    plt.tight_layout()
    comp_plot_path = os.path.join(config.PLOTS_DIR, "model_comparison.png")
    plt.savefig(comp_plot_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()


def evaluate_models() -> Dict[str, Any]:
    """
    Evaluates available trained checkpoints on the untouched test dataset.
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
