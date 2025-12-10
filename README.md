# Graph Attention Networks

A comprehensive, production-ready implementation of Graph Attention Networks with advanced visualization, analysis tools, and interactive demos.

## Features

- **Modern GAT Implementation**: Enhanced Graph Attention Networks with multiple attention mechanisms
- **Comprehensive Evaluation**: Extensive metrics for node classification, link prediction, and graph analysis
- **Attention Visualization**: Interactive tools to analyze and visualize attention patterns
- **Production Ready**: Clean code structure, type hints, comprehensive testing, and documentation
- **Interactive Demo**: Streamlit-based web application for exploring GAT models
- **Configurable Training**: Hydra-based configuration system for easy experimentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Graph-Attention-Networks.git
cd Graph-Attention-Networks

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models import create_model
from src.data import GraphDataset
from src.train import Trainer, get_optimizer, get_scheduler

# Load dataset
dataset = GraphDataset(name="Cora")
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Create model
model = create_model("gat", 
                    in_channels=dataset.get_info()["num_features"],
                    hidden_channels=64,
                    out_channels=dataset.get_info()["num_classes"],
                    heads=8)

# Train model
optimizer = get_optimizer(model, optimizer_name="adam", lr=0.005)
trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer)
test_metrics = trainer.train()
```

### Training with Configuration

```bash
# Train with default configuration
python train.py

# Train with custom configuration
python train.py model.hidden_channels=128 model.heads=16 training.epochs=300

# Train with different dataset
python train.py data.name=CiteSeer training.lr=0.01
```

## Project Structure

```
graph-attention-networks/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── gat.py               # Enhanced GAT implementation
│   │   └── __init__.py          # Model registry
│   ├── data/                     # Data handling
│   │   ├── datasets.py          # Dataset classes
│   │   ├── utils.py             # Data utilities
│   │   └── __init__.py
│   ├── train/                    # Training framework
│   │   ├── trainer.py           # Main trainer class
│   │   ├── utils.py             # Training utilities
│   │   └── __init__.py
│   ├── eval/                     # Evaluation
│   │   ├── metrics.py           # Metrics and analysis
│   │   └── __init__.py
│   └── utils/                    # Utilities
│       └── visualization.py     # Visualization tools
├── configs/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── model/                   # Model configurations
│   ├── data/                    # Data configurations
│   ├── training/                # Training configurations
│   └── evaluation/              # Evaluation configurations
├── demo/                         # Interactive demos
│   └── streamlit_demo.py        # Streamlit web app
├── tests/                        # Test suite
├── assets/                       # Generated assets
├── data/                         # Data storage
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── train.py                      # Main training script
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Model Architecture

### Enhanced GAT Implementation

The implementation includes several enhancements over the original GAT:

- **Multiple Attention Types**: Standard, sparse, and dynamic attention mechanisms
- **Layer Normalization**: Optional layer normalization and batch normalization
- **Residual Connections**: Skip connections for deeper networks
- **Flexible Architecture**: Configurable number of layers, heads, and hidden dimensions
- **Attention Analysis**: Built-in tools for analyzing attention patterns

### Key Features

```python
# Create GAT model with advanced features
model = GAT(
    in_channels=1433,           # Input feature dimension
    hidden_channels=64,         # Hidden layer dimension
    out_channels=7,             # Output classes
    num_layers=2,               # Number of GAT layers
    heads=8,                    # Number of attention heads
    dropout=0.6,                # Dropout rate
    alpha=0.2,                  # LeakyReLU negative slope
    attention_type="standard",   # Attention mechanism type
    use_layer_norm=True,        # Enable layer normalization
    residual_connections=True,  # Enable residual connections
)
```

## Data Handling

### Supported Datasets

- **Citation Networks**: Cora, CiteSeer, PubMed
- **Social Networks**: Custom graph datasets
- **Molecular Graphs**: QM9, ZINC (with RDKit integration)

### Data Augmentation

```python
# Enable data augmentation
dataset = GraphDataset(
    name="Cora",
    augmentation={
        "enabled": True,
        "edge_dropout": 0.1,
        "feature_dropout": 0.1,
        "node_dropout": 0.05
    }
)
```

## Training Framework

### Configuration System

The project uses Hydra for configuration management:

```yaml
# configs/config.yaml
defaults:
  - model: gat
  - data: cora
  - training: default

experiment:
  name: "gat_experiment"
  tags: ["gat", "node_classification"]

device: auto
seed: 42

logging:
  use_wandb: true
  use_tensorboard: true
```

### Training Features

- **Early Stopping**: Configurable patience and monitoring metrics
- **Learning Rate Scheduling**: Multiple scheduler options
- **Gradient Clipping**: Prevent gradient explosion
- **Mixed Precision**: Support for FP16 training
- **Checkpointing**: Automatic model saving and restoration
- **Logging**: Integration with Weights & Biases and TensorBoard

## Evaluation and Metrics

### Comprehensive Metrics

- **Classification Metrics**: Accuracy, F1-score, Precision, Recall, AUROC
- **Attention Analysis**: Attention weight statistics and patterns
- **Model Comparison**: Side-by-side performance comparison
- **Visualization**: Attention heatmaps, network graphs, and distributions

### Usage

```python
from src.eval import MetricsCalculator, AttentionAnalyzer

# Calculate metrics
metrics_calc = MetricsCalculator()
metrics = metrics_calc.calculate_metrics(logits, targets, task="node_classification")

# Analyze attention
analyzer = AttentionAnalyzer(model)
attention_weights = analyzer.get_attention_weights(data)
analysis = analyzer.analyze_attention_patterns(data)
```

## Visualization Tools

### Attention Visualization

```python
from src.utils.visualization import AttentionVisualizer

# Create visualizer
visualizer = AttentionVisualizer(model, data)

# Generate comprehensive report
report = visualizer.generate_comprehensive_report()

# Create specific visualizations
visualizer.plot_attention_heatmap(layer_idx=0, head_idx=0)
visualizer.plot_attention_distribution(layer_idx=0)
visualizer.create_interactive_attention_plot(layer_idx=0, head_idx=0)
```

### Available Visualizations

- **Attention Heatmaps**: 2D heatmaps of attention weights
- **Interactive Networks**: PyVis-based network visualizations
- **Attention Distributions**: Statistical analysis of attention patterns
- **Layer Comparisons**: Cross-layer attention analysis
- **Node Analysis**: Per-node attention pattern exploration

## Interactive Demo

### Streamlit Application

Launch the interactive demo:

```bash
streamlit run demo/streamlit_demo.py
```

### Demo Features

- **Model Upload**: Upload trained models and configurations
- **Interactive Exploration**: Explore attention patterns across layers and heads
- **Real-time Visualization**: Dynamic attention heatmaps and network graphs
- **Node Analysis**: Detailed analysis of individual nodes
- **Metrics Dashboard**: Comprehensive performance metrics
- **Export Options**: Download visualizations and analysis results

## Configuration Reference

### Model Configuration

```yaml
# configs/model/gat.yaml
_target_: src.models.gat.GAT

in_channels: null
hidden_channels: 64
out_channels: null
num_layers: 2
heads: 8
dropout: 0.6
alpha: 0.2
attention_type: "standard"
use_layer_norm: false
residual_connections: false
```

### Training Configuration

```yaml
# configs/training/default.yaml
_target_: src.train.trainer.Trainer

epochs: 200
patience: 50
monitor: "val_accuracy"
mode: "max"

optimizer:
  _target_: torch.optim.Adam
  lr: 0.005
  weight_decay: 5e-4

scheduler:
  _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
  mode: "max"
  factor: 0.5
  patience: 20
```

## Advanced Usage

### Custom Attention Mechanisms

```python
# Implement custom attention
class CustomGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super().__init__()
        # Custom attention implementation
        pass

# Register custom model
from src.models import MODEL_REGISTRY
MODEL_REGISTRY["custom_gat"] = CustomGATLayer
```

### Custom Datasets

```python
# Create custom dataset
class CustomGraphDataset(GraphDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom dataset logic
        pass
```

### Custom Metrics

```python
# Add custom metrics
class CustomMetricsCalculator(MetricsCalculator):
    def calculate_custom_metric(self, logits, targets):
        # Custom metric implementation
        return metric_value
```

## Performance Optimization

### GPU Acceleration

```python
# Automatic device selection
from src.data import get_device
device = get_device()  # CUDA, MPS, or CPU

# Mixed precision training
trainer = Trainer(
    model=model,
    # ... other parameters
    precision=16  # FP16 training
)
```

### Memory Optimization

```python
# Gradient accumulation
trainer = Trainer(
    model=model,
    # ... other parameters
    accumulate_grad_batches=4  # Accumulate gradients
)

# Gradient checkpointing
model = torch.utils.checkpoint.checkpoint_sequential(
    model.layers, 4, input_data
)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py::test_gat_model
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting
black src/
ruff src/
mypy src/
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{graph_attention_networks,
  title={Graph Attention Networks - Modern Implementation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Graph-Attention-Networks}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original GAT paper: Veličković et al. (2018)
- PyTorch Geometric team for the excellent framework
- Streamlit team for the interactive demo framework
- The open-source community for various tools and libraries

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Import Errors**: Ensure all dependencies are installed correctly
3. **Configuration Errors**: Check YAML syntax and parameter names
4. **Visualization Issues**: Install additional dependencies (pyvis, plotly)

### Getting Help

- Check the documentation and examples
- Open an issue on GitHub
- Review the test cases for usage examples

## Roadmap

- [ ] Support for heterogeneous graphs
- [ ] Temporal graph attention networks
- [ ] Graph transformer implementation
- [ ] Distributed training support
- [ ] Model serving with FastAPI
- [ ] Additional visualization options
- [ ] Benchmark suite
- [ ] Tutorial notebooks
# Graph-Attention-Networks
