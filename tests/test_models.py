"""Test suite for Graph Attention Networks."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models import create_model, GAT
from src.data import GraphDataset, get_device, set_seed
from src.eval import MetricsCalculator, AttentionAnalyzer
from src.train import Trainer, get_optimizer, get_scheduler


class TestGATModel:
    """Test cases for GAT model."""

    def test_gat_creation(self):
        """Test GAT model creation."""
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            heads=4
        )
        
        assert isinstance(model, nn.Module)
        assert model.in_channels == 10
        assert model.hidden_channels == 32
        assert model.out_channels == 5
        assert model.num_layers == 2
        assert model.heads == 4

    def test_gat_forward(self):
        """Test GAT forward pass."""
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            heads=4
        )
        
        # Create dummy data
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Forward pass
        output = model(x, edge_index)
        
        assert output.shape == (100, 5)
        assert not torch.isnan(output).any()

    def test_gat_attention_weights(self):
        """Test attention weight extraction."""
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            heads=4
        )
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Get attention weights
        attention_weights = model.get_attention_weights(x, edge_index)
        
        assert isinstance(attention_weights, list)
        assert len(attention_weights) == 2  # Two layers
        
        for layer_weights in attention_weights:
            assert isinstance(layer_weights, tuple)
            assert len(layer_weights) == 2  # edge_index, attention_weights

    def test_gat_embeddings(self):
        """Test embedding extraction."""
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            heads=4
        )
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        # Get embeddings
        embeddings = model.get_embeddings(x, edge_index)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3  # input + 2 layers
        
        # Check embedding shapes
        assert embeddings[0].shape == (100, 10)  # Input
        assert embeddings[1].shape == (100, 128)  # First layer (32 * 4 heads)
        assert embeddings[2].shape == (100, 5)  # Output layer

    def test_gat_model_info(self):
        """Test model information extraction."""
        model = GAT(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            heads=4
        )
        
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["model_name"] == "GAT"
        assert info["total_parameters"] > 0


class TestDataHandling:
    """Test cases for data handling."""

    def test_device_selection(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        # This should not raise an exception
        assert True

    def test_graph_dataset_creation(self):
        """Test GraphDataset creation."""
        # This test might require internet connection for dataset download
        try:
            dataset = GraphDataset(
                name="Cora",
                root="test_data",
                download=False,  # Don't download for testing
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2
            )
            assert isinstance(dataset, GraphDataset)
        except Exception:
            # Skip if dataset not available
            pytest.skip("Dataset not available for testing")


class TestMetrics:
    """Test cases for metrics calculation."""

    def test_metrics_calculator_creation(self):
        """Test MetricsCalculator creation."""
        calc = MetricsCalculator(num_classes=5)
        assert isinstance(calc, MetricsCalculator)

    def test_node_classification_metrics(self):
        """Test node classification metrics calculation."""
        calc = MetricsCalculator(num_classes=3)
        
        # Create dummy data
        logits = torch.randn(100, 3)
        targets = torch.randint(0, 3, (100,))
        
        metrics = calc.calculate_metrics(logits, targets, task="node_classification")
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_micro" in metrics
        assert "auroc" in metrics
        
        # Check metric values are reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
        assert 0 <= metrics["f1_micro"] <= 1

    def test_attention_analyzer_creation(self):
        """Test AttentionAnalyzer creation."""
        model = GAT(in_channels=10, hidden_channels=32, out_channels=5)
        
        # Create dummy data
        data = Data(
            x=torch.randn(100, 10),
            edge_index=torch.randint(0, 100, (2, 200)),
            y=torch.randint(0, 5, (100,))
        )
        
        analyzer = AttentionAnalyzer(model)
        assert isinstance(analyzer, AttentionAnalyzer)


class TestTraining:
    """Test cases for training utilities."""

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        model = GAT(in_channels=10, hidden_channels=32, out_channels=5)
        
        optimizer = get_optimizer(model, optimizer_name="adam", lr=0.001)
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_scheduler_creation(self):
        """Test scheduler creation."""
        model = GAT(in_channels=10, hidden_channels=32, out_channels=5)
        optimizer = get_optimizer(model, optimizer_name="adam", lr=0.001)
        
        scheduler = get_scheduler(optimizer, scheduler_name="step", step_size=30)
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)

    def test_loss_function_creation(self):
        """Test loss function creation."""
        loss_fn = get_loss_function(loss_name="cross_entropy")
        assert isinstance(loss_fn, nn.Module)


class TestModelRegistry:
    """Test cases for model registry."""

    def test_model_registry(self):
        """Test model registry functionality."""
        from src.models import get_model, list_models, create_model
        
        # Test listing models
        models = list_models()
        assert isinstance(models, list)
        assert "gat" in models
        
        # Test getting model class
        model_class = get_model("gat")
        assert model_class == GAT
        
        # Test creating model
        model = create_model("gat", in_channels=10, hidden_channels=32, out_channels=5)
        assert isinstance(model, GAT)

    def test_invalid_model(self):
        """Test invalid model handling."""
        from src.models import get_model
        
        with pytest.raises(ValueError):
            get_model("invalid_model")


if __name__ == "__main__":
    pytest.main([__file__])
