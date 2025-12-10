"""Simple example script for Graph Attention Networks."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

from src.data import get_device, set_seed
from src.models import GAT
from src.eval import MetricsCalculator


def simple_gat_example():
    """Simple GAT example using the original implementation style."""
    print("Running Simple GAT Example")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Cora dataset
    print("Loading Cora dataset...")
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0].to(device)
    
    print(f"Dataset: {dataset.name}")
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")
    print(f"Features: {data.num_node_features}")
    print(f"Classes: {data.num_classes}")
    
    # Create simple GAT model (original style)
    class SimpleGAT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
            super(SimpleGAT, self).__init__()
            self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
            self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.gat1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.gat2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    # Create model
    model = SimpleGAT(
        data.num_node_features, 
        8, 
        data.num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training function
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # Evaluation function
    def test():
        model.eval()
        logits = model(data)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs
    
    # Training loop
    print("\nStarting training...")
    best_test_acc = 0
    
    for epoch in range(1, 201):
        loss = train()
        train_acc, val_acc, test_acc = test()
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}: "
                  f"Loss: {loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Test Acc: {test_acc:.4f}")
    
    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    
    # Calculate comprehensive metrics
    print("\nCalculating comprehensive metrics...")
    model.eval()
    with torch.no_grad():
        logits = model(data)
        
        metrics_calc = MetricsCalculator(num_classes=data.num_classes)
        test_metrics = metrics_calc.calculate_metrics(
            logits[data.test_mask],
            data.y[data.test_mask],
            task="node_classification"
        )
        
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")


def modern_gat_example():
    """Modern GAT example using the enhanced implementation."""
    print("\nRunning Modern GAT Example")
    print("=" * 40)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Cora dataset
    print("Loading Cora dataset...")
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0].to(device)
    
    print(f"Dataset: {dataset.name}")
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")
    print(f"Features: {data.num_node_features}")
    print(f"Classes: {data.num_classes}")
    
    # Create modern GAT model
    model = GAT(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=data.num_classes,
        num_layers=2,
        heads=8,
        dropout=0.6,
        alpha=0.2,
        attention_type="standard",
        use_layer_norm=True,
        residual_connections=True,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Get model info
    model_info = model.get_model_info()
    print(f"\nModel: {model_info['model_name']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Trainable: {model_info['trainable_parameters']:,}")
    print(f"Attention Type: {model_info['attention_type']}")
    print(f"Residual Connections: {model_info['residual_connections']}")
    
    # Training function
    def train():
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # Evaluation function
    def test():
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                pred = logits[mask].argmax(dim=1)
                acc = (pred == data.y[mask]).float().mean().item()
                accs.append(acc)
        return accs
    
    # Training loop
    print("\nStarting training...")
    best_test_acc = 0
    
    for epoch in range(1, 201):
        loss = train()
        train_acc, val_acc, test_acc = test()
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}: "
                  f"Loss: {loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Test Acc: {test_acc:.4f}")
    
    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    
    # Calculate comprehensive metrics
    print("\nCalculating comprehensive metrics...")
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        
        metrics_calc = MetricsCalculator(num_classes=data.num_classes)
        test_metrics = metrics_calc.calculate_metrics(
            logits[data.test_mask],
            data.y[data.test_mask],
            task="node_classification"
        )
        
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Analyze attention patterns
    print("\nAnalyzing attention patterns...")
    from src.eval import AttentionAnalyzer
    
    analyzer = AttentionAnalyzer(model)
    attention_weights = analyzer.get_attention_weights(data)
    
    print(f"Number of layers with attention weights: {len(attention_weights)}")
    
    for i, (edge_index, attn_weights) in enumerate(attention_weights):
        print(f"Layer {i}:")
        print(f"  Edge count: {edge_index.size(1)}")
        print(f"  Attention shape: {attn_weights.shape}")
        print(f"  Mean attention: {attn_weights.mean():.4f}")
        print(f"  Max attention: {attn_weights.max():.4f}")


if __name__ == "__main__":
    print("Graph Attention Networks Examples")
    print("=" * 50)
    
    # Run simple example
    simple_gat_example()
    
    # Run modern example
    modern_gat_example()
    
    print("\nExamples completed successfully!")
    print("For more advanced usage, see the main training script: python train.py")
    print("For interactive exploration, run: streamlit run demo/streamlit_demo.py")
