"""Training utilities and configuration."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR,
)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Get optimizer by name.

    Args:
        model: Model to optimize.
        optimizer_name: Name of the optimizer.
        lr: Learning rate.
        weight_decay: Weight decay.
        **kwargs: Additional optimizer parameters.

    Returns:
        Optimizer instance.
    """
    optimizers = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    optimizer_class = optimizers[optimizer_name]
    return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "reduce_on_plateau",
    **kwargs: Any,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler by name.

    Args:
        optimizer: Optimizer to schedule.
        scheduler_name: Name of the scheduler.
        **kwargs: Additional scheduler parameters.

    Returns:
        Scheduler instance or None.
    """
    if scheduler_name is None or scheduler_name == "none":
        return None
    
    schedulers = {
        "reduce_on_plateau": ReduceLROnPlateau,
        "step": StepLR,
        "exponential": ExponentialLR,
        "cosine": CosineAnnealingLR,
    }
    
    if scheduler_name not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    scheduler_class = schedulers[scheduler_name]
    return scheduler_class(optimizer, **kwargs)


def get_loss_function(
    loss_name: str = "cross_entropy",
    **kwargs: Any,
) -> nn.Module:
    """Get loss function by name.

    Args:
        loss_name: Name of the loss function.
        **kwargs: Additional loss function parameters.

    Returns:
        Loss function instance.
    """
    loss_functions = {
        "cross_entropy": nn.CrossEntropyLoss,
        "nll_loss": nn.NLLLoss,
        "mse_loss": nn.MSELoss,
        "bce_loss": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    loss_class = loss_functions[loss_name]
    return loss_class(**kwargs)


def create_training_config(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create comprehensive training configuration.

    Args:
        model_config: Model configuration.
        data_config: Data configuration.
        training_config: Training configuration.

    Returns:
        Combined training configuration.
    """
    config = {
        "model": model_config,
        "data": data_config,
        "training": training_config,
    }
    
    # Add device configuration
    config["device"] = "auto"
    
    # Add logging configuration
    config["logging"] = {
        "use_wandb": False,
        "use_tensorboard": True,
        "log_dir": "logs",
        "project_name": "gat-experiment",
    }
    
    # Add checkpointing configuration
    config["checkpointing"] = {
        "checkpoint_dir": "checkpoints",
        "save_top_k": 3,
        "save_last": True,
        "every_n_epochs": 10,
    }
    
    return config
