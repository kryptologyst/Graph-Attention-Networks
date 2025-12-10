"""Test configuration and utilities."""

import pytest
import torch
import tempfile
import os
from pathlib import Path

# Set up test environment
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_data():
    """Create sample graph data for testing."""
    return {
        'x': torch.randn(100, 10),
        'edge_index': torch.randint(0, 100, (2, 200)),
        'y': torch.randint(0, 5, (100,)),
        'train_mask': torch.zeros(100, dtype=torch.bool),
        'val_mask': torch.zeros(100, dtype=torch.bool),
        'test_mask': torch.zeros(100, dtype=torch.bool),
    }

@pytest.fixture
def sample_model():
    """Create sample GAT model for testing."""
    from src.models import GAT
    return GAT(
        in_channels=10,
        hidden_channels=32,
        out_channels=5,
        num_layers=2,
        heads=4
    )

# Test configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

# Skip tests that require external dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Skip tests that require internet connection
        if "download" in item.name or "internet" in item.name:
            item.add_marker(pytest.mark.skip(reason="Requires internet connection"))
        
        # Mark slow tests
        if "slow" in item.name or "comprehensive" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
