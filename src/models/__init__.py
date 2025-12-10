"""Model registry and factory functions."""

from typing import Any, Dict, Type

from .gat import GAT

# Model registry
MODEL_REGISTRY: Dict[str, Type] = {
    "gat": GAT,
}


def get_model(model_name: str) -> Type:
    """Get model class by name.

    Args:
        model_name: Name of the model.

    Returns:
        Model class.

    Raises:
        ValueError: If model name is not found.
    """
    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return MODEL_REGISTRY[model_name]


def list_models() -> list[str]:
    """List all available models.

    Returns:
        List of available model names.
    """
    return list(MODEL_REGISTRY.keys())


def create_model(model_name: str, **kwargs: Any) -> Any:
    """Create model instance.

    Args:
        model_name: Name of the model.
        **kwargs: Model initialization parameters.

    Returns:
        Model instance.
    """
    model_class = get_model(model_name)
    return model_class(**kwargs)
