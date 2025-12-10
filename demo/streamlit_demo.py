"""Interactive Streamlit demo for Graph Attention Networks."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from pyvis.network import Network

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import GraphDataset, get_device, set_seed
from src.models import create_model
from src.eval import AttentionAnalyzer, MetricsCalculator
from src.utils.visualization import AttentionVisualizer


def load_model_and_data(model_path: str, config_path: str) -> Tuple[Any, Any, Dict]:
    """Load trained model and data.

    Args:
        model_path: Path to model checkpoint.
        config_path: Path to config file.

    Returns:
        Tuple of (model, data, config).
    """
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    dataset = GraphDataset(
        name=config['data']['name'],
        root=config['data']['root'],
        download=False,
    )
    data = dataset.get_data()
    
    # Load model
    model_config = config['model']
    model_config['in_channels'] = data.num_node_features
    model_config['out_channels'] = data.num_classes
    
    model = create_model("gat", **model_config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, data, config


def create_attention_heatmap(
    attention_weights: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    layer_idx: int,
    head_idx: int,
    max_nodes: int = 100,
) -> go.Figure:
    """Create attention heatmap visualization.

    Args:
        attention_weights: Attention weights tensor.
        edge_index: Edge index tensor.
        num_nodes: Number of nodes in the graph.
        layer_idx: Layer index.
        head_idx: Head index.
        max_nodes: Maximum nodes to display.

    Returns:
        Plotly figure with heatmap.
    """
    # Create adjacency matrix
    adj_matrix = torch.zeros(min(num_nodes, max_nodes), min(num_nodes, max_nodes))
    
    for i, (src, dst) in enumerate(edge_index.t()):
        if src < max_nodes and dst < max_nodes:
            adj_matrix[src, dst] = attention_weights[i]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=adj_matrix.numpy(),
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Source: %{y}<br>Target: %{x}<br>Attention: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}',
        xaxis_title='Target Node',
        yaxis_title='Source Node',
        width=600,
        height=500
    )
    
    return fig


def create_attention_network(
    attention_weights: torch.Tensor,
    edge_index: torch.Tensor,
    node_labels: Optional[torch.Tensor] = None,
    top_k: int = 50,
    threshold: float = 0.1,
) -> Network:
    """Create interactive network visualization.

    Args:
        attention_weights: Attention weights tensor.
        edge_index: Edge index tensor.
        node_labels: Node labels for coloring.
        top_k: Number of top edges to show.
        threshold: Minimum attention threshold.

    Returns:
        PyVis Network object.
    """
    # Filter edges by attention weight
    mask = attention_weights > threshold
    filtered_edges = edge_index[:, mask]
    filtered_weights = attention_weights[mask]
    
    # Get top-k edges
    if len(filtered_weights) > top_k:
        top_k_indices = torch.topk(filtered_weights, top_k)[1]
        filtered_edges = filtered_edges[:, top_k_indices]
        filtered_weights = filtered_weights[top_k_indices]
    
    # Create PyVis network
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes
    nodes = set(filtered_edges[0].tolist()) | set(filtered_edges[1].tolist())
    
    # Color nodes by labels if available
    if node_labels is not None:
        colors = px.colors.qualitative.Set3
        for node in nodes:
            if node < len(node_labels):
                label = node_labels[node].item()
                color = colors[label % len(colors)]
                net.add_node(node, label=str(node), color=color,
                           title=f"Node {node}, Label: {label}")
            else:
                net.add_node(node, label=str(node), color="#97c2fc",
                           title=f"Node {node}")
    else:
        for node in nodes:
            net.add_node(node, label=str(node), color="#97c2fc",
                       title=f"Node {node}")
    
    # Add edges with attention weights
    for i, (src, dst) in enumerate(filtered_edges.t()):
        weight = filtered_weights[i].item()
        net.add_edge(src.item(), dst.item(),
                    value=weight*10,  # Scale for visibility
                    title=f"Attention: {weight:.4f}")
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    return net


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Graph Attention Networks Demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Graph Attention Networks Interactive Demo")
    st.markdown("Explore attention mechanisms in Graph Neural Networks")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_path = st.sidebar.file_uploader(
        "Upload Model Checkpoint",
        type=['pt', 'pth'],
        help="Upload a trained GAT model checkpoint"
    )
    
    config_path = st.sidebar.file_uploader(
        "Upload Config File",
        type=['yaml', 'yml'],
        help="Upload the corresponding config file"
    )
    
    if model_path is None or config_path is None:
        st.warning("Please upload both model checkpoint and config files to start the demo.")
        st.markdown("""
        ### How to use this demo:
        1. Train a GAT model using the training script
        2. Upload the model checkpoint (.pt/.pth file)
        3. Upload the config file (.yaml/.yml file)
        4. Explore the attention mechanisms interactively
        
        ### Features:
        - **Attention Heatmaps**: Visualize attention weights as heatmaps
        - **Interactive Networks**: Explore attention patterns in network graphs
        - **Node Analysis**: Analyze attention patterns for specific nodes
        - **Layer Comparison**: Compare attention across different layers
        - **Metrics Dashboard**: View model performance metrics
        """)
        return
    
    # Load model and data
    try:
        with st.spinner("Loading model and data..."):
            # Save uploaded files temporarily
            temp_model_path = f"temp_model_{model_path.name}"
            temp_config_path = f"temp_config_{config_path.name}"
            
            with open(temp_model_path, "wb") as f:
                f.write(model_path.getbuffer())
            with open(temp_config_path, "wb") as f:
                f.write(config_path.getbuffer())
            
            model, data, config = load_model_and_data(temp_model_path, temp_config_path)
            
            # Clean up temp files
            os.remove(temp_model_path)
            os.remove(temp_config_path)
            
        st.success("Model and data loaded successfully!")
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Get device
    device = get_device()
    model = model.to(device)
    data = data.to(device)
    
    # Sidebar controls
    st.sidebar.subheader("Visualization Controls")
    
    # Layer selection
    num_layers = len([k for k in config['model'].keys() if 'layer' in k.lower()]) or 2
    layer_idx = st.sidebar.selectbox(
        "Select Layer",
        range(num_layers),
        help="Choose which GAT layer to visualize"
    )
    
    # Head selection
    heads = config['model'].get('heads', 8)
    head_idx = st.sidebar.selectbox(
        "Select Attention Head",
        range(heads),
        help="Choose which attention head to visualize"
    )
    
    # Visualization parameters
    max_nodes = st.sidebar.slider(
        "Max Nodes to Display",
        min_value=10,
        max_value=min(500, data.num_nodes),
        value=100,
        help="Maximum number of nodes to show in visualizations"
    )
    
    attention_threshold = st.sidebar.slider(
        "Attention Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Minimum attention weight to display edges"
    )
    
    top_k_edges = st.sidebar.slider(
        "Top-K Edges",
        min_value=10,
        max_value=200,
        value=50,
        help="Number of top attention edges to display"
    )
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Attention Heatmap",
        "🕸️ Interactive Network",
        "🔍 Node Analysis",
        "📈 Layer Comparison",
        "📋 Metrics Dashboard"
    ])
    
    with tab1:
        st.header("Attention Heatmap Visualization")
        
        # Get attention weights
        analyzer = AttentionAnalyzer(model)
        attention_data = analyzer.get_attention_weights(data)
        
        if f"layer_{layer_idx}" in attention_data:
            layer_data = attention_data[f"layer_{layer_idx}"]
            edge_index = layer_data["edge_index"]
            attn_weights = layer_data["attention_weights"]
            
            # Select specific head
            if attn_weights.ndim > 1:
                if head_idx < attn_weights.size(1):
                    attn_weights = attn_weights[:, head_idx]
                else:
                    st.warning(f"Head {head_idx} not available in layer {layer_idx}")
                    attn_weights = attn_weights[:, 0]
            
            # Create heatmap
            fig = create_attention_heatmap(
                attn_weights, edge_index, data.num_nodes,
                layer_idx, head_idx, max_nodes
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Attention statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Attention", f"{attn_weights.mean():.4f}")
            with col2:
                st.metric("Max Attention", f"{attn_weights.max():.4f}")
            with col3:
                st.metric("Min Attention", f"{attn_weights.min():.4f}")
            with col4:
                st.metric("Std Attention", f"{attn_weights.std():.4f}")
        else:
            st.warning(f"Layer {layer_idx} not found in attention weights")
    
    with tab2:
        st.header("Interactive Network Visualization")
        
        if f"layer_{layer_idx}" in attention_data:
            layer_data = attention_data[f"layer_{layer_idx}"]
            edge_index = layer_data["edge_index"]
            attn_weights = layer_data["attention_weights"]
            
            # Select specific head
            if attn_weights.ndim > 1:
                if head_idx < attn_weights.size(1):
                    attn_weights = attn_weights[:, head_idx]
                else:
                    attn_weights = attn_weights[:, 0]
            
            # Create network
            net = create_attention_network(
                attn_weights, edge_index, data.y,
                top_k_edges, attention_threshold
            )
            
            # Save and display network
            net_html = net.generate_html()
            st.components.v1.html(net_html, height=600)
            
            # Download button
            st.download_button(
                label="Download Network HTML",
                data=net_html,
                file_name=f"attention_network_layer_{layer_idx}_head_{head_idx}.html",
                mime="text/html"
            )
        else:
            st.warning(f"Layer {layer_idx} not found in attention weights")
    
    with tab3:
        st.header("Node Analysis")
        
        # Node selection
        node_id = st.selectbox(
            "Select Node to Analyze",
            range(min(100, data.num_nodes)),
            help="Choose a node to analyze its attention patterns"
        )
        
        if f"layer_{layer_idx}" in attention_data:
            layer_data = attention_data[f"layer_{layer_idx}"]
            edge_index = layer_data["edge_index"]
            attn_weights = layer_data["attention_weights"]
            
            # Select specific head
            if attn_weights.ndim > 1:
                if head_idx < attn_weights.size(1):
                    attn_weights = attn_weights[:, head_idx]
                else:
                    attn_weights = attn_weights[:, 0]
            
            # Find edges connected to selected node
            incoming_mask = edge_index[1] == node_id
            outgoing_mask = edge_index[0] == node_id
            
            incoming_edges = edge_index[:, incoming_mask]
            outgoing_edges = edge_index[:, outgoing_mask]
            incoming_weights = attn_weights[incoming_mask]
            outgoing_weights = attn_weights[outgoing_mask]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Incoming Attention (Node {node_id})")
                if len(incoming_weights) > 0:
                    incoming_df = pd.DataFrame({
                        'Source Node': incoming_edges[0].tolist(),
                        'Attention Weight': incoming_weights.tolist()
                    })
                    st.dataframe(incoming_df.sort_values('Attention Weight', ascending=False))
                else:
                    st.info("No incoming edges found")
            
            with col2:
                st.subheader(f"Outgoing Attention (Node {node_id})")
                if len(outgoing_weights) > 0:
                    outgoing_df = pd.DataFrame({
                        'Target Node': outgoing_edges[1].tolist(),
                        'Attention Weight': outgoing_weights.tolist()
                    })
                    st.dataframe(outgoing_df.sort_values('Attention Weight', ascending=False))
                else:
                    st.info("No outgoing edges found")
            
            # Node statistics
            if hasattr(data, 'y') and data.y is not None:
                node_label = data.y[node_id].item()
                st.info(f"Node {node_id} has label: {node_label}")
        else:
            st.warning(f"Layer {layer_idx} not found in attention weights")
    
    with tab4:
        st.header("Layer Comparison")
        
        # Compare attention patterns across layers
        if len(attention_data) > 1:
            layers_to_compare = st.multiselect(
                "Select Layers to Compare",
                list(attention_data.keys()),
                default=list(attention_data.keys())[:2]
            )
            
            if len(layers_to_compare) >= 2:
                # Create comparison plot
                fig = make_subplots(
                    rows=1, cols=len(layers_to_compare),
                    subplot_titles=[f"Layer {i}" for i in layers_to_compare]
                )
                
                for i, layer_key in enumerate(layers_to_compare):
                    layer_data = attention_data[layer_key]
                    attn_weights = layer_data["attention_weights"]
                    
                    # Use first head for comparison
                    if attn_weights.ndim > 1:
                        attn_weights = attn_weights[:, 0]
                    
                    fig.add_trace(
                        go.Histogram(
                            x=attn_weights.numpy(),
                            name=f"Layer {layer_key.split('_')[1]}",
                            opacity=0.7
                        ),
                        row=1, col=i+1
                    )
                
                fig.update_layout(
                    title="Attention Weight Distribution Comparison",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least 2 layers for comparison")
        else:
            st.info("Only one layer available for comparison")
    
    with tab5:
        st.header("Metrics Dashboard")
        
        # Calculate metrics
        metrics_calculator = MetricsCalculator()
        
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            
            # Test metrics
            test_metrics = metrics_calculator.calculate_metrics(
                logits[data.test_mask],
                data.y[data.test_mask],
                task="node_classification"
            )
            
            # Validation metrics
            val_metrics = metrics_calculator.calculate_metrics(
                logits[data.val_mask],
                data.y[data.val_mask],
                task="node_classification"
            )
            
            # Training metrics
            train_metrics = metrics_calculator.calculate_metrics(
                logits[data.train_mask],
                data.y[data.train_mask],
                task="node_classification"
            )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Training Metrics")
            for metric, value in train_metrics.items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
        
        with col2:
            st.subheader("Validation Metrics")
            for metric, value in val_metrics.items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
        
        with col3:
            st.subheader("Test Metrics")
            for metric, value in test_metrics.items():
                st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
        
        # Model information
        st.subheader("Model Information")
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            info_df = pd.DataFrame(list(model_info.items()), columns=['Property', 'Value'])
            st.dataframe(info_df, use_container_width=True)
        
        # Dataset information
        st.subheader("Dataset Information")
        dataset_info = {
            'Property': ['Nodes', 'Edges', 'Features', 'Classes', 'Train Nodes', 'Val Nodes', 'Test Nodes'],
            'Value': [
                data.num_nodes,
                data.num_edges,
                data.num_node_features,
                data.num_classes,
                data.train_mask.sum().item(),
                data.val_mask.sum().item(),
                data.test_mask.sum().item()
            ]
        }
        dataset_df = pd.DataFrame(dataset_info)
        st.dataframe(dataset_df, use_container_width=True)


if __name__ == "__main__":
    main()
