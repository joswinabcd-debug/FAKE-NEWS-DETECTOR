"""
LSTM Model Architecture for TRUTHSCAN AI.
Implements Long Short-Term Memory recurrent neural network for sequence-based text classification.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM Classifier Architecture:
    Input Text (Batch, Seq_Len)
        ↓
    Embedding Layer (Batch, Seq_Len, Embed_Dim)
        ↓
    LSTM Layer (Embedding -> Hidden Representation)
        ↓
    Final Hidden State Extraction (h_n of final recurrent layer)
        ↓
    Dropout (0.3)
        ↓
    Fully Connected Layer -> (Batch, 1)
        ↓
    Binary Classification Logits
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        
        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # Standard unidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Regularization Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Dense classification projection
        self.fc = nn.Linear(hidden_dim, 1)
        
        # Parameter Initialization
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier best practices."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.padding_idx is not None and self.padding_idx >= 0:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
                
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)
                # Set forget gate bias to 1.0 (standard LSTM practice for long-term memory)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
                
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, sequence_length)
        Returns:
            logits: Tensor of shape (batch_size, 1)
        """
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM output:
        # out: (batch_size, seq_len, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        # c_n: (num_layers, batch_size, hidden_dim)
        out, (h_n, c_n) = self.lstm(embedded)
        
        # Extract the true final hidden state at the last non-padding token position
        mask = (x != self.padding_idx)
        lengths = mask.sum(dim=1)
        lengths = torch.clamp(lengths, min=1)
        batch_idx = torch.arange(x.size(0), device=x.device)
        last_hidden = out[batch_idx, lengths - 1]
        
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)  # (batch_size, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to get calibrated sigmoid probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
