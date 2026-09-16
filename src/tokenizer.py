"""
Tokenizer and Vocabulary Management for TRUTHSCAN AI.
Builds and maintains word-to-index mappings, sequence truncation, and padding.
"""

import os
import re
import json
from collections import Counter
from typing import List, Dict, Optional
import config


class Tokenizer:
    def __init__(self, max_vocab_size: int = config.MAX_VOCAB_SIZE, 
                 max_sequence_length: int = config.MAX_SEQUENCE_LENGTH):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.word2idx: Dict[str, int] = {
            config.PAD_TOKEN: config.PAD_IDX,
            config.UNK_TOKEN: config.UNK_IDX
        }
        self.idx2word: Dict[int, str] = {
            config.PAD_IDX: config.PAD_TOKEN,
            config.UNK_IDX: config.UNK_TOKEN
        }
        self.is_fitted = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Splits text into lowercased tokens using regex word boundaries.
        Preserves alphanumeric tokens and basic contractions.
        """
        if not text:
            return []
        tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
        return tokens

    def fit_on_texts(self, texts: List[str]) -> None:
        """
        Builds vocabulary strictly from training texts.
        Caps vocabulary at max_vocab_size including <PAD> and <UNK>.
        """
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Space reserved for <PAD> (0) and <UNK> (1)
        available_slots = self.max_vocab_size - 2
        most_common = counter.most_common(available_slots)

        self.word2idx = {
            config.PAD_TOKEN: config.PAD_IDX,
            config.UNK_TOKEN: config.UNK_IDX
        }
        self.idx2word = {
            config.PAD_IDX: config.PAD_TOKEN,
            config.UNK_IDX: config.UNK_TOKEN
        }

        current_idx = 2
        for word, _ in most_common:
            self.word2idx[word] = current_idx
            self.idx2word[current_idx] = word
            current_idx += 1

        self.is_fitted = True

    def encode(self, text: str) -> List[int]:
        """
        Transforms text into fixed-length list of token indices with truncation and padding.
        """
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(token, config.UNK_IDX) for token in tokens]

        # Truncate
        if len(indices) > self.max_sequence_length:
            indices = indices[:self.max_sequence_length]

        # Pad
        if len(indices) < self.max_sequence_length:
            indices = indices + [config.PAD_IDX] * (self.max_sequence_length - len(indices))

        return indices

    def decode(self, indices: List[int]) -> List[str]:
        """
        Converts token IDs back to words, excluding padding.
        """
        return [self.idx2word.get(idx, config.UNK_TOKEN) 
                for idx in indices if idx != config.PAD_IDX]

    def get_vocab_size(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: str = config.VOCABULARY_FILE) -> None:
        """Saves the vocabulary mapping to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "max_vocab_size": self.max_vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "word2idx": self.word2idx,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str = config.VOCABULARY_FILE) -> "Tokenizer":
        """Loads vocabulary mapping from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Vocabulary file not found at {filepath}. "
                f"Please train the model first to generate vocabulary."
            )
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            max_vocab_size=data.get("max_vocab_size", config.MAX_VOCAB_SIZE),
            max_sequence_length=data.get("max_sequence_length", config.MAX_SEQUENCE_LENGTH)
        )
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() 
                              else idx: word for word, idx in tokenizer.word2idx.items()}
        # Ensure reversed lookup is correct
        tokenizer.idx2word = {v: k for k, v in tokenizer.word2idx.items()}
        tokenizer.is_fitted = True
        return tokenizer
