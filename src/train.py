"""
Training Pipeline for TRUTHSCAN AI.
Supports TextCNN, LSTM, or Both with validation tracking, checkpointing,
learning rate scheduling, and progress reporting.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional, Callable

import config
from src.dataset import prepare_data, set_seed, check_dataset_exists
from src.tokenizer import Tokenizer
from models.textcnn_model import TextCNN
from models.lstm_model import LSTMClassifier


def train_single_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, float]:
    """Trains a model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataloader)

    for i, (sequences, labels) in enumerate(dataloader):
        sequences = sequences.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if progress_callback and (i % max(1, num_batches // 10) == 0 or i == num_batches - 1):
            batch_pct = (i + 1) / num_batches
            epoch_progress = ((epoch - 1) + batch_pct) / total_epochs
            status = f"Epoch {epoch}/{total_epochs} - Batch {i+1}/{num_batches} - Loss: {loss.item():.4f}"
            progress_callback(epoch_progress, status)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return {"loss": epoch_loss, "acc": epoch_acc}


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates model performance on the validation split."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(sequences)
            loss = criterion(logits, labels)

            running_loss += loss.item() * sequences.size(0)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return {"loss": val_loss, "acc": val_acc}


def train_model(
    model_type: str = "cnn",
    mode: str = "quick",
    data_dict: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Tokenizer] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Complete training orchestrator for a specific model architecture.
    
    Args:
        model_type: 'cnn' or 'lstm'
        mode: 'quick', 'normal', or 'full'
        data_dict: pre-prepared data loaders dictionary (or prepared automatically if None)
        tokenizer: Tokenizer instance
        progress_callback: callback for overall progress (0.0 - 1.0) and status string
        log_callback: callback for appending log text lines
    
    Returns:
        Dictionary containing training history, best validation accuracy, checkpoint path, and training time.
    """
    model_type = model_type.lower()
    if model_type not in ["cnn", "lstm"]:
        raise ValueError("model_type must be either 'cnn' or 'lstm'")

    exists, err_msg = check_dataset_exists()
    if not exists:
        raise FileNotFoundError(err_msg)

    set_seed(config.RANDOM_SEED)
    device = config.DEVICE

    def log(msg: str):
        print(msg)
        if log_callback:
            log_callback(msg)

    log(f"==================================================")
    log(f"Starting Training for: {model_type.upper()}")
    log(f"Mode: {mode.upper()} | Device: {device}")
    log(f"==================================================")

    # Prepare data if not supplied
    if data_dict is None or tokenizer is None:
        log("Preparing dataset and building vocabulary from training data...")
        data_dict, tokenizer = prepare_data(mode=mode)

    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]
    vocab_size = tokenizer.get_vocab_size()

    # Training settings
    mode_cfg = config.TRAINING_MODES.get(mode, config.TRAINING_MODES["quick"])
    epochs = mode_cfg["epochs"]
    lr = mode_cfg["learning_rate"]

    # Instantiate model
    if model_type == "cnn":
        model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=config.CNN_CONFIG["embedding_dim"],
            num_filters=config.CNN_CONFIG["num_filters"],
            kernel_sizes=config.CNN_CONFIG["kernel_sizes"],
            dropout=config.CNN_CONFIG["dropout"],
            padding_idx=config.PAD_IDX
        ).to(device)
        checkpoint_path = config.CNN_CHECKPOINT
        history_path = config.CNN_HISTORY_FILE
    else:
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.LSTM_CONFIG["embedding_dim"],
            hidden_dim=config.LSTM_CONFIG["hidden_dim"],
            num_layers=config.LSTM_CONFIG["num_layers"],
            dropout=config.LSTM_CONFIG["dropout"],
            padding_idx=config.PAD_IDX
        ).to(device)
        checkpoint_path = config.LSTM_CHECKPOINT
        history_path = config.LSTM_HISTORY_FILE

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": epochs,
        "mode": mode,
        "device": str(device)
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_res = train_single_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            progress_callback=progress_callback
        )
        
        val_res = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step(val_res["loss"])

        epoch_duration = time.time() - epoch_start
        history["train_loss"].append(round(train_res["loss"], 4))
        history["train_acc"].append(round(train_res["acc"], 4))
        history["val_loss"].append(round(val_res["loss"], 4))
        history["val_acc"].append(round(val_res["acc"], 4))

        log_line = (
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] - "
            f"Train Loss: {train_res['loss']:.4f} | Train Acc: {train_res['acc'] * 100:.2f}% | "
            f"Val Loss: {val_res['loss']:.4f} | Val Acc: {val_res['acc'] * 100:.2f}%"
        )
        log(log_line)

        # Save checkpoint strictly if validation performance improves
        if val_res["acc"] > best_val_acc or (val_res["acc"] == best_val_acc and val_res["loss"] < best_val_loss):
            best_val_acc = val_res["acc"]
            best_val_loss = val_res["loss"]
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "val_loss": best_val_loss,
                "vocab_size": vocab_size,
                "model_type": model_type,
                "config": config.CNN_CONFIG if model_type == "cnn" else config.LSTM_CONFIG
            }, checkpoint_path)
            
            log(f"--> Saved new best checkpoint to {checkpoint_path} (Val Acc: {best_val_acc * 100:.2f}%)")

    total_training_time = time.time() - start_time
    history["total_training_time"] = round(total_training_time, 2)
    history["best_val_acc"] = round(best_val_acc, 4)

    # Persist history JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    log(f"Training completed for {model_type.upper()} in {total_training_time:.1f}s. Best Val Acc: {best_val_acc * 100:.2f}%\n")

    return {
        "model_type": model_type,
        "history": history,
        "best_val_acc": best_val_acc,
        "checkpoint_path": checkpoint_path,
        "training_time": total_training_time
    }


def train_pipeline(
    target: str = "both",
    mode: str = "quick",
    progress_callback: Optional[Callable[[float, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    High-level orchestrator to train 'cnn', 'lstm', or 'both'.
    Reuses the single pre-split dataset and vocabulary across both models to ensure identical evaluation splits.
    """
    target = target.lower()
    if target not in ["cnn", "lstm", "both"]:
        raise ValueError("target must be 'cnn', 'lstm', or 'both'")

    if log_callback:
        log_callback("Preparing stratified dataset splits with sanitized text...")
    data_dict, tokenizer = prepare_data(mode=mode)

    results = {}
    if target in ["cnn", "both"]:
        def cnn_callback(p, s):
            if progress_callback:
                progress_callback(p * 0.5 if target == "both" else p, f"[CNN] {s}")
        results["cnn"] = train_model(
            "cnn", mode=mode, data_dict=data_dict, tokenizer=tokenizer,
            progress_callback=cnn_callback, log_callback=log_callback
        )

    if target in ["lstm", "both"]:
        def lstm_callback(p, s):
            if progress_callback:
                progress_callback(0.5 + p * 0.5 if target == "both" else p, f"[LSTM] {s}")
        results["lstm"] = train_model(
            "lstm", mode=mode, data_dict=data_dict, tokenizer=tokenizer,
            progress_callback=lstm_callback, log_callback=log_callback
        )

    return results
