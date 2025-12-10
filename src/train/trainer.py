"""Training utilities and trainer class."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from ..data import get_device, set_seed
from ..eval.metrics import MetricsCalculator


class EarlyStopping:
    """Early stopping utility."""

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.001,
        monitor: str = "val_accuracy",
        mode: str = "max",
        restore_best_weights: bool = True,
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            monitor: Metric to monitor.
            mode: 'max' for maximizing, 'min' for minimizing.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric score.
            model: Model to potentially restore weights.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self.restore_checkpoint(model)

        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()

    def restore_checkpoint(self, model: nn.Module) -> None:
        """Restore model checkpoint."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class Trainer:
    """Main trainer class for GNN models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        device: Optional[torch.device] = None,
        epochs: int = 200,
        patience: int = 50,
        min_delta: float = 0.001,
        monitor: str = "val_accuracy",
        mode: str = "max",
        gradient_clip_val: Optional[float] = None,
        accumulate_grad_batches: int = 1,
        precision: int = 32,
        log_every_n_steps: int = 10,
        val_check_interval: float = 1.0,
        check_val_every_n_epoch: int = 1,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        log_dir: str = "logs",
        project_name: str = "gat-experiment",
        checkpoint_dir: str = "checkpoints",
        save_top_k: int = 3,
        save_last: bool = True,
        every_n_epochs: int = 10,
        filename: str = "epoch_{epoch:03d}-val_acc_{val_accuracy:.4f}",
    ) -> None:
        """Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            loss_fn: Loss function.
            device: Device to use.
            epochs: Number of training epochs.
            patience: Early stopping patience.
            min_delta: Early stopping minimum delta.
            monitor: Metric to monitor for early stopping.
            mode: Mode for early stopping ('max' or 'min').
            gradient_clip_val: Gradient clipping value.
            accumulate_grad_batches: Number of batches to accumulate gradients.
            precision: Training precision (16, 32, bf16).
            log_every_n_steps: Log every n steps.
            val_check_interval: Validation check interval.
            check_val_every_n_epoch: Check validation every n epochs.
            use_wandb: Whether to use Weights & Biases.
            use_tensorboard: Whether to use TensorBoard.
            log_dir: Logging directory.
            project_name: Project name for logging.
            checkpoint_dir: Checkpoint directory.
            save_top_k: Number of best checkpoints to save.
            save_last: Whether to save last checkpoint.
            every_n_epochs: Save checkpoint every n epochs.
            filename: Checkpoint filename template.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device or get_device()
        
        # Training configuration
        self.epochs = epochs
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            monitor=monitor,
            mode=mode,
        )
        
        # Logging
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.project_name = project_name
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.filename = filename
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize metrics
        self.metrics_calculator = MetricsCalculator()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = None
        self.checkpoint_history = []

    def _setup_logging(self) -> None:
        """Setup logging."""
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # TensorBoard
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(self.log_dir)
        
        # Weights & Biases
        if self.use_wandb:
            wandb.init(project=self.project_name)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Get training data
        train_data = next(iter(self.train_loader))
        train_data = train_data.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if hasattr(self.model, 'forward'):
            logits = self.model(train_data.x, train_data.edge_index)
        else:
            logits = self.model(train_data)
        
        # Compute loss
        loss = self.loss_fn(logits[train_data.train_mask], train_data.y[train_data.train_mask])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        
        self.optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Calculate training metrics
        with torch.no_grad():
            pred = logits[train_data.train_mask].argmax(dim=1)
            train_acc = (pred == train_data.y[train_data.train_mask]).float().mean().item()
        
        return {
            "train_loss": total_loss / num_batches,
            "train_accuracy": train_acc,
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        
        # Get validation data
        val_data = next(iter(self.val_loader))
        val_data = val_data.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                logits = self.model(val_data.x, val_data.edge_index)
            else:
                logits = self.model(val_data)
            
            # Compute loss
            loss = self.loss_fn(logits[val_data.val_mask], val_data.y[val_data.val_mask])
            
            # Calculate metrics
            pred = logits[val_data.val_mask].argmax(dim=1)
            val_acc = (pred == val_data.y[val_data.val_mask]).float().mean().item()
            
            # Calculate additional metrics
            metrics = self.metrics_calculator.calculate_metrics(
                logits[val_data.val_mask],
                val_data.y[val_data.val_mask],
                task="node_classification"
            )
        
        return {
            "val_loss": loss.item(),
            "val_accuracy": val_acc,
            **metrics,
        }

    def test(self) -> Dict[str, float]:
        """Test the model.

        Returns:
            Dictionary with test metrics.
        """
        self.model.eval()
        
        # Get test data
        test_data = next(iter(self.test_loader))
        test_data = test_data.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                logits = self.model(test_data.x, test_data.edge_index)
            else:
                logits = self.model(test_data)
            
            # Compute loss
            loss = self.loss_fn(logits[test_data.test_mask], test_data.y[test_data.test_mask])
            
            # Calculate metrics
            pred = logits[test_data.test_mask].argmax(dim=1)
            test_acc = (pred == test_data.y[test_data.test_mask]).float().mean().item()
            
            # Calculate additional metrics
            metrics = self.metrics_calculator.calculate_metrics(
                logits[test_data.test_mask],
                test_data.y[test_data.test_mask],
                task="node_classification"
            )
        
        return {
            "test_loss": loss.item(),
            "test_accuracy": test_acc,
            **metrics,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            metrics: Current metrics.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save checkpoint
        filename = self.filename.format(epoch=epoch, **metrics)
        filepath = os.path.join(self.checkpoint_dir, f"{filename}.pt")
        torch.save(checkpoint, filepath)
        
        # Update checkpoint history
        self.checkpoint_history.append({
            "epoch": epoch,
            "filepath": filepath,
            "metrics": metrics,
        })
        
        # Keep only top-k checkpoints
        if len(self.checkpoint_history) > self.save_top_k:
            # Sort by monitored metric
            monitor_key = self.early_stopping.monitor
            self.checkpoint_history.sort(
                key=lambda x: x["metrics"].get(monitor_key, 0),
                reverse=self.early_stopping.mode == "max"
            )
            
            # Remove worst checkpoint
            worst_checkpoint = self.checkpoint_history.pop()
            if os.path.exists(worst_checkpoint["filepath"]):
                os.remove(worst_checkpoint["filepath"])

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to various backends.

        Args:
            metrics: Metrics to log.
            step: Current step.
        """
        # TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
        
        # Weights & Biases
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def train(self) -> Dict[str, float]:
        """Main training loop.

        Returns:
            Dictionary with final test metrics.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            if epoch % self.check_val_every_n_epoch == 0:
                val_metrics = self.validate()
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                
                # Log metrics
                self.log_metrics(all_metrics, epoch)
                
                # Print progress
                print(f"Epoch {epoch:03d}: "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Val Acc: {val_metrics['val_accuracy']:.4f}")
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics[self.early_stopping.monitor])
                    else:
                        self.scheduler.step()
                
                # Early stopping
                monitor_value = val_metrics[self.early_stopping.monitor]
                if self.early_stopping(monitor_value, self.model):
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                # Save checkpoint
                if epoch % self.every_n_epochs == 0:
                    self.save_checkpoint(epoch, all_metrics)
            
            self.global_step += 1
        
        # Final test
        print("Running final test...")
        test_metrics = self.test()
        
        # Log final metrics
        self.log_metrics(test_metrics, self.current_epoch)
        
        # Print final results
        print("\nFinal Results:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Close logging
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()
        
        return test_metrics
