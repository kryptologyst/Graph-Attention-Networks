"""Evaluation metrics and utilities."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
import torchmetrics
from torchmetrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    AUROC,
    AveragePrecision,
)


class MetricsCalculator:
    """Comprehensive metrics calculator for graph tasks."""

    def __init__(self, num_classes: Optional[int] = None) -> None:
        """Initialize metrics calculator.

        Args:
            num_classes: Number of classes for multi-class tasks.
        """
        self.num_classes = num_classes
        
        # Initialize torchmetrics
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.f1_micro = F1Score(task="multiclass", num_classes=num_classes, average="micro")
        self.precision_macro = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.precision_micro = Precision(task="multiclass", num_classes=num_classes, average="micro")
        self.recall_macro = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.recall_micro = Recall(task="multiclass", num_classes=num_classes, average="micro")
        self.auroc = AUROC(task="multiclass", num_classes=num_classes)

    def calculate_metrics(
        self,
        logits: Tensor,
        targets: Tensor,
        task: str = "node_classification",
        return_dict: bool = True,
    ) -> Union[Dict[str, float], Tuple[float, ...]]:
        """Calculate comprehensive metrics.

        Args:
            logits: Model predictions/logits.
            targets: Ground truth targets.
            task: Task type ('node_classification', 'link_prediction', 'graph_classification').
            return_dict: Whether to return metrics as dictionary.

        Returns:
            Metrics dictionary or tuple of metric values.
        """
        if task == "node_classification":
            return self._calculate_node_classification_metrics(logits, targets, return_dict)
        elif task == "link_prediction":
            return self._calculate_link_prediction_metrics(logits, targets, return_dict)
        elif task == "graph_classification":
            return self._calculate_graph_classification_metrics(logits, targets, return_dict)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _calculate_node_classification_metrics(
        self,
        logits: Tensor,
        targets: Tensor,
        return_dict: bool = True,
    ) -> Union[Dict[str, float], Tuple[float, ...]]:
        """Calculate node classification metrics.

        Args:
            logits: Model predictions/logits.
            targets: Ground truth targets.
            return_dict: Whether to return metrics as dictionary.

        Returns:
            Metrics dictionary or tuple of metric values.
        """
        # Convert to numpy for sklearn metrics
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy()
        else:
            logits_np = logits
        
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy()
        else:
            targets_np = targets

        # Get predictions
        if logits_np.ndim > 1:
            predictions = np.argmax(logits_np, axis=1)
        else:
            predictions = (logits_np > 0.5).astype(int)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(targets_np, predictions),
            "f1_macro": f1_score(targets_np, predictions, average="macro", zero_division=0),
            "f1_micro": f1_score(targets_np, predictions, average="micro", zero_division=0),
            "precision_macro": precision_score(targets_np, predictions, average="macro", zero_division=0),
            "precision_micro": precision_score(targets_np, predictions, average="micro", zero_division=0),
            "recall_macro": recall_score(targets_np, predictions, average="macro", zero_division=0),
            "recall_micro": recall_score(targets_np, predictions, average="micro", zero_division=0),
        }

        # Calculate AUROC if binary or multi-class
        try:
            if len(np.unique(targets_np)) == 2:
                # Binary classification
                if logits_np.ndim > 1:
                    metrics["auroc"] = roc_auc_score(targets_np, logits_np[:, 1])
                else:
                    metrics["auroc"] = roc_auc_score(targets_np, logits_np)
            else:
                # Multi-class classification
                metrics["auroc"] = roc_auc_score(targets_np, logits_np, multi_class="ovr", average="macro")
        except ValueError:
            metrics["auroc"] = 0.0

        if return_dict:
            return metrics
        else:
            return tuple(metrics.values())

    def _calculate_link_prediction_metrics(
        self,
        logits: Tensor,
        targets: Tensor,
        return_dict: bool = True,
    ) -> Union[Dict[str, float], Tuple[float, ...]]:
        """Calculate link prediction metrics.

        Args:
            logits: Model predictions/logits.
            targets: Ground truth targets.
            return_dict: Whether to return metrics as dictionary.

        Returns:
            Metrics dictionary or tuple of metric values.
        """
        # Convert to numpy
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy()
        else:
            logits_np = logits
        
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy()
        else:
            targets_np = targets

        # Get predictions
        if logits_np.ndim > 1:
            predictions = (logits_np[:, 1] > 0.5).astype(int)
            scores = logits_np[:, 1]
        else:
            predictions = (logits_np > 0.5).astype(int)
            scores = logits_np

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(targets_np, predictions),
            "f1_macro": f1_score(targets_np, predictions, average="macro", zero_division=0),
            "f1_micro": f1_score(targets_np, predictions, average="micro", zero_division=0),
            "precision_macro": precision_score(targets_np, predictions, average="macro", zero_division=0),
            "precision_micro": precision_score(targets_np, predictions, average="micro", zero_division=0),
            "recall_macro": recall_score(targets_np, predictions, average="macro", zero_division=0),
            "recall_micro": recall_score(targets_np, predictions, average="micro", zero_division=0),
        }

        # Calculate AUROC and Average Precision
        try:
            metrics["auroc"] = roc_auc_score(targets_np, scores)
            metrics["average_precision"] = average_precision_score(targets_np, scores)
        except ValueError:
            metrics["auroc"] = 0.0
            metrics["average_precision"] = 0.0

        if return_dict:
            return metrics
        else:
            return tuple(metrics.values())

    def _calculate_graph_classification_metrics(
        self,
        logits: Tensor,
        targets: Tensor,
        return_dict: bool = True,
    ) -> Union[Dict[str, float], Tuple[float, ...]]:
        """Calculate graph classification metrics.

        Args:
            logits: Model predictions/logits.
            targets: Ground truth targets.
            return_dict: Whether to return metrics as dictionary.

        Returns:
            Metrics dictionary or tuple of metric values.
        """
        # Use the same metrics as node classification
        return self._calculate_node_classification_metrics(logits, targets, return_dict)


class AttentionAnalyzer:
    """Analyzer for attention weights in GAT models."""

    def __init__(self, model: Any) -> None:
        """Initialize attention analyzer.

        Args:
            model: GAT model with attention weights.
        """
        self.model = model

    def get_attention_weights(
        self,
        data: Any,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get attention weights from the model.

        Args:
            data: Graph data.
            layer_idx: Specific layer index (None for all layers).
            head_idx: Specific head index (None for all heads).

        Returns:
            Dictionary with attention weights.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get attention weights
            if hasattr(self.model, 'get_attention_weights'):
                attention_weights = self.model.get_attention_weights(
                    data.x, data.edge_index, data.edge_attr
                )
            else:
                # Fallback: forward pass with attention weights
                _, attention_weights = self.model(
                    data.x, data.edge_index, data.edge_attr, return_attention_weights=True
                )

        # Process attention weights
        processed_weights = {}
        
        for i, (edge_index, attn_weights) in enumerate(attention_weights):
            layer_key = f"layer_{i}"
            processed_weights[layer_key] = {
                "edge_index": edge_index,
                "attention_weights": attn_weights,
                "num_heads": attn_weights.size(1) if attn_weights.ndim > 1 else 1,
            }

        return processed_weights

    def analyze_attention_patterns(
        self,
        data: Any,
        top_k: int = 10,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Analyze attention patterns.

        Args:
            data: Graph data.
            top_k: Number of top attention weights to analyze.
            threshold: Minimum attention weight threshold.

        Returns:
            Dictionary with attention analysis.
        """
        attention_weights = self.get_attention_weights(data)
        
        analysis = {}
        
        for layer_key, layer_data in attention_weights.items():
            edge_index = layer_data["edge_index"]
            attn_weights = layer_data["attention_weights"]
            
            # Calculate attention statistics
            if attn_weights.ndim > 1:
                # Multi-head attention
                mean_attention = attn_weights.mean(dim=1)
                max_attention = attn_weights.max(dim=1)[0]
                std_attention = attn_weights.std(dim=1)
            else:
                # Single-head attention
                mean_attention = attn_weights
                max_attention = attn_weights
                std_attention = torch.zeros_like(attn_weights)

            # Find top-k attention weights
            top_k_indices = torch.topk(mean_attention, min(top_k, len(mean_attention)))[1]
            
            # Find edges above threshold
            above_threshold = mean_attention > threshold
            
            analysis[layer_key] = {
                "mean_attention": mean_attention,
                "max_attention": max_attention,
                "std_attention": std_attention,
                "top_k_indices": top_k_indices,
                "above_threshold": above_threshold,
                "top_k_edges": edge_index[:, top_k_indices],
                "top_k_weights": mean_attention[top_k_indices],
                "attention_statistics": {
                    "mean": mean_attention.mean().item(),
                    "std": mean_attention.std().item(),
                    "min": mean_attention.min().item(),
                    "max": mean_attention.max().item(),
                    "above_threshold_count": above_threshold.sum().item(),
                    "above_threshold_ratio": above_threshold.float().mean().item(),
                }
            }

        return analysis

    def visualize_attention_heatmap(
        self,
        data: Any,
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create attention heatmap visualization data.

        Args:
            data: Graph data.
            layer_idx: Layer index.
            head_idx: Head index.
            save_path: Path to save visualization.

        Returns:
            Dictionary with visualization data.
        """
        attention_weights = self.get_attention_weights(data)
        
        if f"layer_{layer_idx}" not in attention_weights:
            raise ValueError(f"Layer {layer_idx} not found in attention weights")
        
        layer_data = attention_weights[f"layer_{layer_idx}"]
        edge_index = layer_data["edge_index"]
        attn_weights = layer_data["attention_weights"]
        
        # Select specific head if multi-head
        if attn_weights.ndim > 1:
            if head_idx >= attn_weights.size(1):
                raise ValueError(f"Head {head_idx} not found in layer {layer_idx}")
            attn_weights = attn_weights[:, head_idx]
        
        # Create adjacency matrix for visualization
        num_nodes = data.num_nodes
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        
        for i, (src, dst) in enumerate(edge_index.t()):
            adj_matrix[src, dst] = attn_weights[i]
        
        visualization_data = {
            "adjacency_matrix": adj_matrix.tolist(),
            "attention_weights": attn_weights.tolist(),
            "edge_index": edge_index.tolist(),
            "num_nodes": num_nodes,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
        }
        
        return visualization_data
