"""
test_models.py

Tests for src/models/neural_network.py: forward-pass output shape,
embedding-extraction shape, and that extract_embeddings + classifier
compose correctly to produce the same logits as forward().
"""
import torch
import pytest

from models.neural_network import NeuralClassifier


@pytest.fixture
def model():
    return NeuralClassifier(
        input_dim=20,
        hidden_dimensions=[16, 8],
        embedding_dimension=4,
        num_classes=5,
        dropout=0.1,
    )


def test_forward_pass_output_shape(model):
    model.eval()
    x = torch.randn(10, 20)
    logits = model(x)
    assert logits.shape == (10, 5)


def test_embedding_extraction_shape(model):
    model.eval()
    x = torch.randn(10, 20)
    embeddings = model.extract_embeddings(x)
    assert embeddings.shape == (10, 4)


def test_forward_uses_extracted_embeddings(model):
    """forward(x) must equal classifier(extract_embeddings(x)) — i.e. the
    embedding layer is genuinely on the path to the classifier, not a
    disconnected side branch."""
    model.eval()
    x = torch.randn(6, 20)
    with torch.no_grad():
        logits_direct = model(x)
        embeddings = model.extract_embeddings(x)
        logits_via_embedding = model.classifier(embeddings)
    assert torch.allclose(logits_direct, logits_via_embedding, atol=1e-6)


def test_single_sample_batch_with_eval_mode(model):
    """BatchNorm requires batch size > 1 in train mode; a lone sample should
    still work in eval mode (the mode inference scripts actually use)."""
    model.eval()
    x = torch.randn(1, 20)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 5)