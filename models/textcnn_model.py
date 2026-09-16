"""
TextCNN Model Architecture for TRUTHSCAN AI.
Implements multi-filter 1D convolutional neural network for text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TextCNN(nn.Module):
    """
    TextCNN Classifier Architecture:
    Input Text (Batch, Seq_Len)
        ↓
    Embedding Layer (Batch, Seq_Len, Embed_Dim) -> Permute to (Batch, Embed_Dim, Seq_Len)
        ↓
    Multiple Conv1D Layers (Kernels: 3, 4, 5, Filters: 100 each)
        ↓
    ReLU Activation
        ↓
    Max-over-time Pooling (Batch, Filters, 1) -> (Batch, Filters)
        ↓
    Concatenate pooled outputs -> (Batch, len(Kernels) * Filters)
        ↓
    Dropout (0.5)
        ↓
    Fully Connected Layer -> (Batch, 1)
        ↓
    Binary Classification Logits
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        super(TextCNN, self).__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
            
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = list(kernel_sizes)
        self.padding_idx = padding_idx
        
        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # Parallel 1D Convolutions with distinct receptive fields (e.g. 3, 4, 5)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k
            ) for k in self.kernel_sizes
        ])
        
        # Regularization Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Dense classification projection
        total_filter_units = num_filters * len(self.kernel_sizes)
        self.fc = nn.Linear(total_filter_units, 1)
        
        # Parameter Initialization
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier/Kaiming best practices."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.padding_idx is not None and self.padding_idx >= 0:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
                
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)
                
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
        
        # Conv1D expects (batch_size, in_channels, seq_len)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        
        # Guard against inputs shorter than the maximum kernel size
        max_k = max(self.kernel_sizes)
        if embedded.size(2) < max_k:
            pad_amount = max_k - embedded.size(2)
            embedded = F.pad(embedded, (0, pad_amount))
        
        # Apply convolution, ReLU, and global max pooling over time
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, L_out)
            # Global adaptive max pooling over the temporal dimension
            pooled = F.adaptive_max_pool1d(conv_out, 1).view(conv_out.size(0), self.num_filters)
            pooled_outputs.append(pooled)
            
        # Concatenate multi-scale feature maps
        cat = torch.cat(pooled_outputs, dim=1)  # (batch_size, total_filter_units)
        cat = self.dropout(cat)
        
        logits = self.fc(cat)  # (batch_size, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to get calibrated sigmoid probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
