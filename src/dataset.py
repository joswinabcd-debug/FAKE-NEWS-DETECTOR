"""
Dataset Loading, Preprocessing, Splitting, and PyTorch Dataset Definition for TRUTHSCAN AI.
Strictly adheres to ISOT dataset structure, stratified 80/10/10 split, and zero-leakage tokenization.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional

import config
from src.preprocessing import combine_title_and_text, clean_text
from src.tokenizer import Tokenizer


# Enforce reproducibility across libraries
def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NewsDataset(Dataset):
    """
    PyTorch Dataset yielding pre-encoded token tensor and binary label.
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def load_file_dataframe(filepath: str) -> pd.DataFrame:
    """Reads a dataset file whether it is an Excel (.xlsx, .xls) or CSV (.csv)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath)
    else:
        return pd.read_csv(filepath)


def check_dataset_exists() -> Tuple[bool, str]:
    """
    Verifies presence of Fake and True dataset files in archive/ or data/.
    Returns (exists, error_message).
    """
    fake_path = config.find_dataset_file("Fake")
    true_path = config.find_dataset_file("True")

    missing = []
    if not (os.path.exists(fake_path) and os.path.getsize(fake_path) > 0):
        missing.append("Fake dataset (Fake.csv or Fake.xlsx)")
    if not (os.path.exists(true_path) and os.path.getsize(true_path) > 0):
        missing.append("True dataset (True.csv or True.xlsx)")

    if missing:
        msg = (
            "Dataset not found.\n\n"
            "Please place:\n"
            "data/Fake.csv (or archive/Fake.csv)\n"
            "data/True.csv (or archive/True.csv)"
        )
        return False, msg
    return True, ""


def load_raw_data() -> pd.DataFrame:
    """
    Loads Fake and True news datasets (CSV or Excel), validates columns,
    assigns labels (Fake=1, True=0), handles missing values, and removes duplicates.
    """
    exists, err_msg = check_dataset_exists()
    if not exists:
        raise FileNotFoundError(err_msg)

    fake_path = config.find_dataset_file("Fake")
    true_path = config.find_dataset_file("True")

    expected_cols = {"title", "text", "subject", "date"}

    # Load Fake articles (label = 1)
    df_fake = load_file_dataframe(fake_path)
    # Standardize lower-cased column names
    df_fake.columns = [c.strip().lower() for c in df_fake.columns]
    missing_fake = expected_cols - set(df_fake.columns)
    if missing_fake:
        raise ValueError(f"Fake dataset ({os.path.basename(fake_path)}) is missing required columns: {sorted(list(missing_fake))}")
    df_fake["label"] = 1

    # Load True articles (label = 0)
    df_true = load_file_dataframe(true_path)
    # Standardize lower-cased column names
    df_true.columns = [c.strip().lower() for c in df_true.columns]
    missing_true = expected_cols - set(df_true.columns)
    if missing_true:
        raise ValueError(f"True dataset ({os.path.basename(true_path)}) is missing required columns: {sorted(list(missing_true))}")
    df_true["label"] = 0

    # Combine datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # Handle missing values
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["subject"] = df["subject"].fillna("unknown")
    df["date"] = df["date"].fillna("unknown")

    # Combine title and text for model input
    df["full_text"] = df.apply(lambda r: combine_title_and_text(str(r["title"]), str(r["text"])), axis=1)

    # Remove empty articles
    df = df[df["full_text"].str.strip() != ""].copy()

    # Remove duplicates
    df = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)

    return df


_DATASET_STATS_CACHE: Optional[Dict[str, Any]] = None

def get_dataset_statistics() -> Dict[str, Any]:
    """
    Computes genuine statistical metrics from the active ISOT dataset.
    Caches results to disk/memory to ensure sub-millisecond API responses.
    Returns error metadata if files are absent.
    """
    global _DATASET_STATS_CACHE
    if _DATASET_STATS_CACHE is not None:
        return _DATASET_STATS_CACHE

    stats_cache_path = os.path.join(config.DATA_DIR, "dataset_stats.json")
    if os.path.exists(stats_cache_path):
        try:
            with open(stats_cache_path, "r", encoding="utf-8") as f:
                _DATASET_STATS_CACHE = json.load(f)
                return _DATASET_STATS_CACHE
        except Exception:
            pass

    exists, err_msg = check_dataset_exists()
    if not exists:
        return {"available": False, "error": err_msg}

    try:
        df = load_raw_data()
        total_articles = len(df)
        fake_count = int((df["label"] == 1).sum())
        real_count = int((df["label"] == 0).sum())

        # Word lengths
        word_counts = df["full_text"].apply(lambda t: len(t.split()))
        avg_length = float(word_counts.mean()) if total_articles > 0 else 0.0

        # Split projections
        train_samples = int(total_articles * config.TRAIN_RATIO)
        val_samples = int(total_articles * config.VAL_RATIO)
        test_samples = total_articles - train_samples - val_samples

        res = {
            "available": True,
            "total_articles": total_articles,
            "fake_articles": fake_count,
            "real_articles": real_count,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "avg_article_length": round(avg_length, 1),
            "subjects": df["subject"].value_counts().to_dict(),
        }
        _DATASET_STATS_CACHE = res
        try:
            with open(stats_cache_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        except Exception:
            pass
        return res
    except Exception as e:
        return {"available": False, "error": str(e)}


def prepare_data(mode: str = "quick", custom_samples: Optional[int] = None) -> Tuple[Dict[str, Any], Tokenizer]:
    """
    Prepares stratified train, val, and test splits.
    Tokenizes strictly on training split to eliminate data leakage.
    Saves vocabulary to disk.
    
    Returns:
        split_data: dictionary with train, val, test datasets and loaders.
        tokenizer: fitted Tokenizer instance.
    """
    set_seed(config.RANDOM_SEED)
    df = load_raw_data()

    # Determine sample count based on mode
    target_samples = custom_samples
    if target_samples is None:
        mode_cfg = config.TRAINING_MODES.get(mode, config.TRAINING_MODES["quick"])
        target_samples = mode_cfg.get("samples")

    # If sampling requested and dataset is larger, perform stratified subsampling
    if target_samples is not None and target_samples < len(df):
        df, _ = train_test_split(
            df,
            train_size=target_samples,
            stratify=df["label"],
            random_state=config.RANDOM_SEED
        )
        df = df.reset_index(drop=True)

    # Stratified Split: 80% train, 20% temp (which splits 50/50 into 10% val, 10% test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=config.RANDOM_SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=config.RANDOM_SEED
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Fit tokenizer ONLY on training texts
    tokenizer = Tokenizer(
        max_vocab_size=config.MAX_VOCAB_SIZE,
        max_sequence_length=config.MAX_SEQUENCE_LENGTH
    )
    tokenizer.fit_on_texts(train_df["full_text"].tolist())
    tokenizer.save(config.VOCABULARY_FILE)

    # Encode texts
    X_train = np.array([tokenizer.encode(t) for t in train_df["full_text"]])
    y_train = train_df["label"].values.astype(np.float32)

    X_val = np.array([tokenizer.encode(t) for t in val_df["full_text"]])
    y_val = val_df["label"].values.astype(np.float32)

    X_test = np.array([tokenizer.encode(t) for t in test_df["full_text"]])
    y_test = test_df["label"].values.astype(np.float32)

    # Persist processed splits for fast reloading
    np.savez_compressed(
        os.path.join(config.PROCESSED_DIR, "splits.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    # Create PyTorch datasets
    train_dataset = NewsDataset(X_train, y_train)
    val_dataset = NewsDataset(X_val, y_val)
    test_dataset = NewsDataset(X_test, y_test)

    batch_size = config.TRAINING_MODES.get(mode, {}).get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "vocab_size": tokenizer.get_vocab_size(),
        "train_fake": int((train_df["label"] == 1).sum()),
        "train_real": int((train_df["label"] == 0).sum()),
        "test_fake": int((test_df["label"] == 1).sum()),
        "test_real": int((test_df["label"] == 0).sum()),
    }

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "metadata": metadata
    }, tokenizer


def load_processed_splits(batch_size: int = 32) -> Optional[Tuple[DataLoader, DataLoader, DataLoader, Tokenizer]]:
    """
    Loads saved processed splits and vocabulary from disk if present.
    """
    splits_path = os.path.join(config.PROCESSED_DIR, "splits.npz")
    if not os.path.exists(splits_path) or not os.path.exists(config.VOCABULARY_FILE):
        return None

    tokenizer = Tokenizer.load(config.VOCABULARY_FILE)
    data = np.load(splits_path)
    
    train_ds = NewsDataset(data["X_train"], data["y_train"])
    val_ds = NewsDataset(data["X_val"], data["y_val"])
    test_ds = NewsDataset(data["X_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer
