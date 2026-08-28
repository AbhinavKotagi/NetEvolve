"""
neural_network.py

Purpose: Feed-forward embedding network for known-class traffic classification.

Architecture (configurable via config.yaml -> neural_model):
    Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x len(hidden_dimensions)
          -> Linear (embedding_dimension)   <- this is the explicit embedding layer
          -> Linear (num_classes)            <- classifier head

The embedding layer is exposed via extract_embeddings(x), which returns the
representation immediately before the classifier head. This is what later
phases (open-set detection, novel class discovery) will consume — the model
never needs to be retrained for those phases.
"""
from typing import List

import torch

import torch.nn as nn


class NeuralClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dimensions: List[int],
        embedding_dimension: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dimensions:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.embedding_layer = nn.Linear(prev_dim, embedding_dimension)
        self.classifier = nn.Linear(embedding_dimension, num_classes)

        self.input_dim = input_dim
        self.embedding_dimension = embedding_dimension
        self.num_classes = num_classes

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 'embedding_dimension'-wide representation immediately
        before the classifier head. Works in both train and eval mode."""
        h = self.hidden_layers(x)
        return self.embedding_layer(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.extract_embeddings(x)
        logits = self.classifier(embedding)
        return logits