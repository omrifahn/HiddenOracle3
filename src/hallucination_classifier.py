"""
hallucination_classifier.py

Defines the hallucination classifier model.
"""

import torch
import torch.nn as nn


class SimpleLinearClassifier(nn.Module):
    """
    A simple linear classifier that predicts
    "0 => factual" or "1 => hallucinating."
    """
    def __init__(self, input_dim: int, num_labels: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        logits = self.linear(x)
        return logits
