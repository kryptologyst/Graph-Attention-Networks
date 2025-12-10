"""Main training script for Graph Attention Networks."""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import GraphDataset, DataLoader, get_device, set_seed
from src.models import create_model
from src.train import Trainer, get_optimizer, get_scheduler, get_loss_function


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    # Set random seed
    set_seed(cfg.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = GraphDataset(
        name=cfg.data.name,
        root=cfg.data.root,
        download=cfg.data.download,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        random_split=cfg.data.random_split,
        stratified_split=cfg.data.stratified_split,
        augmentation=cfg.data.augmentation,
        normalize_features=cfg.data.normalize_features,
        add_self_loops=cfg.data.add_self_loops,
        make_undirected=cfg.data.make_undirected,
        remove_isolated_nodes=cfg.data.remove_isolated_nodes,
    )
    
    # Print dataset info
    dataset_info = dataset.get_info()
    print(f"Dataset: {dataset_info['name']}")
    print(f"Nodes: {dataset_info['num_nodes']:,}")
    print(f"Edges: {dataset_info['num_edges']:,}")
    print(f"Features: {dataset_info['num_features']}")
    print(f"Classes: {dataset_info['num_classes']}")
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False)
    
    # Create model
    print("Creating model...")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config["in_channels"] = dataset_info["num_features"]
    model_config["out_channels"] = dataset_info["num_classes"]
    
    model = create_model("gat", **model_config)
    model = model.to(device)
    
    # Print model info
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"Model: {model_info['model_name']}")
        print(f"Parameters: {model_info['total_parameters']:,}")
        print(f"Trainable: {model_info['trainable_parameters']:,}")
    
    # Create optimizer
    optimizer_config = OmegaConf.to_container(cfg.training.optimizer, resolve=True)
    optimizer = get_optimizer(model, **optimizer_config)
    
    # Create scheduler
    scheduler_config = OmegaConf.to_container(cfg.training.scheduler, resolve=True)
    scheduler = get_scheduler(optimizer, **scheduler_config)
    
    # Create loss function
    loss_config = OmegaConf.to_container(cfg.training.loss, resolve=True)
    loss_fn = get_loss_function(**loss_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        min_delta=cfg.training.min_delta,
        monitor=cfg.training.monitor,
        mode=cfg.training.mode,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        use_wandb=cfg.logging.use_wandb,
        use_tensorboard=cfg.logging.use_tensorboard,
        log_dir=cfg.logging.log_dir,
        project_name=cfg.logging.project_name,
        checkpoint_dir=cfg.paths.checkpoint_dir,
        save_top_k=cfg.training.checkpoint.save_top_k,
        save_last=cfg.training.checkpoint.save_last,
        every_n_epochs=cfg.training.checkpoint.every_n_epochs,
        filename=cfg.training.checkpoint.filename,
    )
    
    # Start training
    print("Starting training...")
    test_metrics = trainer.train()
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save final configuration
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.paths.output_dir, "config.yaml"))
    
    print(f"\nTraining completed! Results saved to {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
