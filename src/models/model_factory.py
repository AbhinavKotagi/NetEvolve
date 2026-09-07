"""
model_factory.py

Purpose: Builds models from config.yaml hyperparameters, so scripts never
construct sklearn/PyTorch models directly with hardcoded settings.
"""
from sklearn.ensemble import RandomForestClassifier


from models.neural_network import NeuralClassifier


def build_baseline_model(config: dict) -> RandomForestClassifier:
    params = config["baseline_model"]["hyperparameters"]
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        class_weight=params["class_weight"],
        n_jobs=params["n_jobs"],
        random_state=config["project"]["random_seed"],
    )


def build_neural_model(config: dict, input_dim: int, num_classes: int) -> NeuralClassifier:
    nm = config["neural_model"]
    return NeuralClassifier(
        input_dim=input_dim,
        hidden_dimensions=nm["hidden_dimensions"],
        embedding_dimension=nm["embedding_dimension"],
        num_classes=num_classes,
        dropout=nm["dropout"],
    )


# ---------------------------------------------------------------------------
# RoNeTC — Phase 2: MultiViewFeatureExtractor factory
# ---------------------------------------------------------------------------

def build_multiview_extractor(config: dict, ip_shape, transport_shape, payload_shape):
    """Build a MultiViewFeatureExtractor from config + per-view shapes.

    Parameters
    ----------
    config : dict
        Full project config.
    ip_shape, transport_shape, payload_shape : (H_p, W_p)
        Per-packet 2D tensor shapes.  Get from ViewEncoder(max_bytes).shape.

    Returns
    -------
    MultiViewFeatureExtractor
    """
    from models.global_local_extractor import MultiViewFeatureExtractor
    return MultiViewFeatureExtractor.from_config(
        config,
        ip_shape=ip_shape,
        transport_shape=transport_shape,
        payload_shape=payload_shape,
    )