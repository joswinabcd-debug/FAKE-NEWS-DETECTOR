"""
Inference & Prediction Module for TRUTHSCAN AI.
Loads saved weights and vocabulary, processes input text, and returns calibrated predictions.
"""

import os
import torch
from typing import Dict, Any, Tuple, Optional

import config
from src.preprocessing import combine_title_and_text
from src.tokenizer import Tokenizer
from src.evaluate import load_model_from_checkpoint


class NewsPredictor:
    """
    Inference orchestrator for TRUTHSCAN AI.
    Loads models on-demand and caches them in memory.
    """
    def __init__(self):
        self.device = config.DEVICE
        self.tokenizer: Optional[Tokenizer] = None
        self.models: Dict[str, torch.nn.Module] = {}

    def is_model_available(self, model_type: str) -> bool:
        """Checks whether model weights and vocabulary exist."""
        if not os.path.exists(config.VOCABULARY_FILE):
            return False
        if model_type.lower() == "cnn":
            return os.path.exists(config.CNN_CHECKPOINT)
        elif model_type.lower() == "lstm":
            return os.path.exists(config.LSTM_CHECKPOINT)
        elif model_type.lower() in ["both", "compare both"]:
            return os.path.exists(config.CNN_CHECKPOINT) and os.path.exists(config.LSTM_CHECKPOINT)
        return False

    def _load_resources(self, model_type: str):
        """Loads tokenizer and requested model into memory."""
        if not os.path.exists(config.VOCABULARY_FILE):
            raise FileNotFoundError(
                "Model not trained yet.\n"
                "Please train the model before making predictions."
            )

        if self.tokenizer is None:
            self.tokenizer = Tokenizer.load(config.VOCABULARY_FILE)

        m_key = model_type.lower()
        if m_key not in self.models:
            if m_key == "cnn":
                if not os.path.exists(config.CNN_CHECKPOINT):
                    raise FileNotFoundError(
                        "CNN model not trained yet.\n"
                        "Please train the CNN model before making predictions."
                    )
                self.models["cnn"] = load_model_from_checkpoint("cnn", config.CNN_CHECKPOINT, self.device)
            elif m_key == "lstm":
                if not os.path.exists(config.LSTM_CHECKPOINT):
                    raise FileNotFoundError(
                        "LSTM model not trained yet.\n"
                        "Please train the LSTM model before making predictions."
                    )
                self.models["lstm"] = load_model_from_checkpoint("lstm", config.LSTM_CHECKPOINT, self.device)

    def predict_single(self, title: str, text: str, model_type: str = "cnn") -> Dict[str, Any]:
        """
        Executes prediction for a single model ('cnn' or 'lstm').
        
        Returns:
            Dictionary with:
                - model: 'CNN' or 'LSTM'
                - prediction: 'REAL NEWS' or 'FAKE NEWS'
                - label_int: 0 (Real) or 1 (Fake)
                - confidence: float (0.0 to 100.0)
                - raw_prob: float (0.0 to 1.0) probability of being fake
                - full_text_length: int (word count)
        """
        combined = combine_title_and_text(title, text)
        if not combined.strip():
            raise ValueError("Please enter a headline or article before analyzing.")

        m_key = model_type.lower()
        self._load_resources(m_key)
        model = self.models[m_key]

        # Encode tokens
        token_ids = self.tokenizer.encode(combined)
        seq_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        model.eval()
        with torch.no_grad():
            logits = model(seq_tensor)
            fake_prob = torch.sigmoid(logits).item()

        if fake_prob >= 0.5:
            prediction = "FAKE NEWS"
            label_int = 1
            confidence = fake_prob * 100.0
        else:
            prediction = "REAL NEWS"
            label_int = 0
            confidence = (1.0 - fake_prob) * 100.0

        return {
            "model": model_type.upper(),
            "prediction": prediction,
            "label_int": label_int,
            "confidence": round(confidence, 2),
            "raw_prob": round(fake_prob, 4),
            "word_count": len(combined.split()),
            "headline_used": title.strip() if title else "(No headline provided)",
        }

    def predict_both(self, title: str, text: str) -> Dict[str, Any]:
        """
        Evaluates input independently on both CNN and LSTM without artificial ensembling.
        """
        combined = combine_title_and_text(title, text)
        if not combined.strip():
            raise ValueError("Please enter a headline or article before analyzing.")

        res_cnn = self.predict_single(title, text, model_type="cnn")
        res_lstm = self.predict_single(title, text, model_type="lstm")

        return {
            "cnn": res_cnn,
            "lstm": res_lstm,
            "comparison": [
                {
                    "Model": "CNN",
                    "Prediction": res_cnn["prediction"],
                    "Confidence": f"{res_cnn['confidence']:.2f}%",
                    "Raw Fake Probability": f"{res_cnn['raw_prob'] * 100:.2f}%"
                },
                {
                    "Model": "LSTM",
                    "Prediction": res_lstm["prediction"],
                    "Confidence": f"{res_lstm['confidence']:.2f}%",
                    "Raw Fake Probability": f"{res_lstm['raw_prob'] * 100:.2f}%"
                }
            ]
        }
