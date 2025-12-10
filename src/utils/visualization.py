"""Attention visualization utilities for GAT models."""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import torch
from torch import Tensor
import networkx as nx
from pyvis.network import Network

from ..data import get_graph_statistics
from ..eval import AttentionAnalyzer


class AttentionVisualizer:
    """Comprehensive attention visualization for GAT models."""

    def __init__(
        self,
        model: Any,
        data: Any,
        save_dir: str = "assets/attention_viz",
        style: str = "seaborn-v0_8",
    ) -> None:
        """Initialize attention visualizer.

        Args:
            model: Trained GAT model.
            data: Graph data.
            save_dir: Directory to save visualizations.
            style: Matplotlib style.
        """
        self.model = model
        self.data = data
        self.save_dir = save_dir
        self.style = style
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use(style)
        
        # Initialize attention analyzer
        self.analyzer = AttentionAnalyzer(model)

    def plot_attention_heatmap(
        self,
        layer_idx: int = 0,
        head_idx: int = 0,
        max_nodes: int = 100,
        figsize: Tuple[int, int] = (12, 10),
        save: bool = True,
    ) -> None:
        """Plot attention heatmap for a specific layer and head.

        Args:
            layer_idx: Layer index.
            head_idx: Head index.
            max_nodes: Maximum number of nodes to display.
            figsize: Figure size.
            save: Whether to save the plot.
        """
        # Get attention weights
        attention_data = self.analyzer.get_attention_weights(self.data)
        
        if f"layer_{layer_idx}" not in attention_data:
            print(f"Layer {layer_idx} not found in attention weights")
            return
        
        layer_data = attention_data[f"layer_{layer_idx}"]
        edge_index = layer_data["edge_index"]
        attn_weights = layer_data["attention_weights"]
        
        # Select specific head
        if attn_weights.ndim > 1:
            if head_idx >= attn_weights.size(1):
                print(f"Head {head_idx} not found in layer {layer_idx}")
                return
            attn_weights = attn_weights[:, head_idx]
        
        # Create adjacency matrix
        num_nodes = min(self.data.num_nodes, max_nodes)
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        
        for i, (src, dst) in enumerate(edge_index.t()):
            if src < num_nodes and dst < num_nodes:
                adj_matrix[src, dst] = attn_weights[i]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(adj_matrix.numpy(), cmap='viridis', aspect='auto')
        
        ax.set_title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}')
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        plt.tight_layout()
        
        if save:
            filename = f"attention_heatmap_layer_{layer_idx}_head_{head_idx}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {filepath}")
        
        plt.show()

    def plot_attention_distribution(
        self,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 8),
        save: bool = True,
    ) -> None:
        """Plot attention weight distribution.

        Args:
            layer_idx: Layer index.
            head_idx: Specific head index (None for all heads).
            figsize: Figure size.
            save: Whether to save the plot.
        """
        attention_data = self.analyzer.get_attention_weights(self.data)
        
        if f"layer_{layer_idx}" not in attention_data:
            print(f"Layer {layer_idx} not found in attention weights")
            return
        
        layer_data = attention_data[f"layer_{layer_idx}"]
        attn_weights = layer_data["attention_weights"]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        if attn_weights.ndim == 1:
            # Single head
            weights = attn_weights.numpy()
            
            # Histogram
            axes[0].hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_title('Attention Weight Distribution')
            axes[0].set_xlabel('Attention Weight')
            axes[0].set_ylabel('Frequency')
            
            # Box plot
            axes[1].boxplot(weights)
            axes[1].set_title('Attention Weight Box Plot')
            axes[1].set_ylabel('Attention Weight')
            
            # Cumulative distribution
            sorted_weights = np.sort(weights)
            cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
            axes[2].plot(sorted_weights, cumulative)
            axes[2].set_title('Cumulative Distribution')
            axes[2].set_xlabel('Attention Weight')
            axes[2].set_ylabel('Cumulative Probability')
            
            # Statistics
            stats_text = f"""
            Mean: {weights.mean():.4f}
            Std: {weights.std():.4f}
            Min: {weights.min():.4f}
            Max: {weights.max():.4f}
            Median: {np.median(weights):.4f}
            """
            axes[3].text(0.1, 0.5, stats_text, transform=axes[3].transAxes, 
                         fontsize=12, verticalalignment='center')
            axes[3].set_title('Statistics')
            axes[3].axis('off')
            
        else:
            # Multi-head
            num_heads = attn_weights.size(1)
            
            for i in range(min(4, num_heads)):
                weights = attn_weights[:, i].numpy()
                
                axes[i].hist(weights, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
                axes[i].set_title(f'Head {i} - Attention Distribution')
                axes[i].set_xlabel('Attention Weight')
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save:
            filename = f"attention_distribution_layer_{layer_idx}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Attention distribution saved to {filepath}")
        
        plt.show()

    def plot_attention_network(
        self,
        layer_idx: int = 0,
        head_idx: int = 0,
        top_k: int = 50,
        threshold: float = 0.1,
        node_size: int = 20,
        figsize: Tuple[int, int] = (15, 12),
        save: bool = True,
    ) -> None:
        """Plot attention network visualization.

        Args:
            layer_idx: Layer index.
            head_idx: Head index.
            top_k: Number of top attention edges to show.
            threshold: Minimum attention threshold.
            node_size: Node size in visualization.
            figsize: Figure size.
            save: Whether to save the plot.
        """
        attention_data = self.analyzer.get_attention_weights(self.data)
        
        if f"layer_{layer_idx}" not in attention_data:
            print(f"Layer {layer_idx} not found in attention weights")
            return
        
        layer_data = attention_data[f"layer_{layer_idx}"]
        edge_index = layer_data["edge_index"]
        attn_weights = layer_data["attention_weights"]
        
        # Select specific head
        if attn_weights.ndim > 1:
            if head_idx >= attn_weights.size(1):
                print(f"Head {head_idx} not found in layer {layer_idx}")
                return
            attn_weights = attn_weights[:, head_idx]
        
        # Filter edges by attention weight
        mask = attn_weights > threshold
        filtered_edges = edge_index[:, mask]
        filtered_weights = attn_weights[mask]
        
        # Get top-k edges
        if len(filtered_weights) > top_k:
            top_k_indices = torch.topk(filtered_weights, top_k)[1]
            filtered_edges = filtered_edges[:, top_k_indices]
            filtered_weights = filtered_weights[top_k_indices]
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        nodes = set(filtered_edges[0].tolist()) | set(filtered_edges[1].tolist())
        G.add_nodes_from(nodes)
        
        # Add edges with attention weights
        for i, (src, dst) in enumerate(filtered_edges.t()):
            G.add_edge(src.item(), dst.item(), weight=filtered_weights[i].item())
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                              node_color='lightblue', alpha=0.7, ax=ax)
        
        # Draw edges with thickness based on attention weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                              alpha=0.6, edge_color='gray', ax=ax)
        
        # Add labels for high-attention nodes
        high_attention_nodes = []
        for node in G.nodes():
            in_weights = [G[pred][node]['weight'] for pred in G.predecessors(node)]
            out_weights = [G[node][succ]['weight'] for succ in G.successors(node)]
            total_weight = sum(in_weights) + sum(out_weights)
            if total_weight > np.percentile(weights, 90):
                high_attention_nodes.append(node)
        
        nx.draw_networkx_labels(G, pos, {n: str(n) for n in high_attention_nodes}, 
                               font_size=8, ax=ax)
        
        ax.set_title(f'Attention Network - Layer {layer_idx}, Head {head_idx}\n'
                    f'Top {len(filtered_weights)} edges (threshold: {threshold})')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            filename = f"attention_network_layer_{layer_idx}_head_{head_idx}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Attention network saved to {filepath}")
        
        plt.show()

    def create_interactive_attention_plot(
        self,
        layer_idx: int = 0,
        head_idx: int = 0,
        max_nodes: int = 100,
        save: bool = True,
    ) -> go.Figure:
        """Create interactive attention visualization using Plotly.

        Args:
            layer_idx: Layer index.
            head_idx: Head index.
            max_nodes: Maximum number of nodes to display.
            save: Whether to save the plot.

        Returns:
            Plotly figure object.
        """
        attention_data = self.analyzer.get_attention_weights(self.data)
        
        if f"layer_{layer_idx}" not in attention_data:
            print(f"Layer {layer_idx} not found in attention weights")
            return None
        
        layer_data = attention_data[f"layer_{layer_idx}"]
        edge_index = layer_data["edge_index"]
        attn_weights = layer_data["attention_weights"]
        
        # Select specific head
        if attn_weights.ndim > 1:
            if head_idx >= attn_weights.size(1):
                print(f"Head {head_idx} not found in layer {layer_idx}")
                return None
            attn_weights = attn_weights[:, head_idx]
        
        # Filter nodes
        num_nodes = min(self.data.num_nodes, max_nodes)
        mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        filtered_edges = edge_index[:, mask]
        filtered_weights = attn_weights[mask]
        
        # Create adjacency matrix
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        for i, (src, dst) in enumerate(filtered_edges.t()):
            adj_matrix[src, dst] = filtered_weights[i]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=adj_matrix.numpy(),
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Source: %{y}<br>Target: %{x}<br>Attention: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Interactive Attention Heatmap - Layer {layer_idx}, Head {head_idx}',
            xaxis_title='Target Node',
            yaxis_title='Source Node',
            width=800,
            height=600
        )
        
        if save:
            filename = f"interactive_attention_layer_{layer_idx}_head_{head_idx}.html"
            filepath = os.path.join(self.save_dir, filename)
            fig.write_html(filepath)
            print(f"Interactive attention plot saved to {filepath}")
        
        return fig

    def create_pyvis_network(
        self,
        layer_idx: int = 0,
        head_idx: int = 0,
        top_k: int = 100,
        threshold: float = 0.05,
        save: bool = True,
    ) -> Network:
        """Create interactive network visualization using PyVis.

        Args:
            layer_idx: Layer index.
            head_idx: Head index.
            top_k: Number of top attention edges to show.
            threshold: Minimum attention threshold.
            save: Whether to save the network.

        Returns:
            PyVis Network object.
        """
        attention_data = self.analyzer.get_attention_weights(self.data)
        
        if f"layer_{layer_idx}" not in attention_data:
            print(f"Layer {layer_idx} not found in attention weights")
            return None
        
        layer_data = attention_data[f"layer_{layer_idx}"]
        edge_index = layer_data["edge_index"]
        attn_weights = layer_data["attention_weights"]
        
        # Select specific head
        if attn_weights.ndim > 1:
            if head_idx >= attn_weights.size(1):
                print(f"Head {head_idx} not found in layer {layer_idx}")
                return None
            attn_weights = attn_weights[:, head_idx]
        
        # Filter edges by attention weight
        mask = attn_weights > threshold
        filtered_edges = edge_index[:, mask]
        filtered_weights = attn_weights[mask]
        
        # Get top-k edges
        if len(filtered_weights) > top_k:
            top_k_indices = torch.topk(filtered_weights, top_k)[1]
            filtered_edges = filtered_edges[:, top_k_indices]
            filtered_weights = filtered_weights[top_k_indices]
        
        # Create PyVis network
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        
        # Add nodes
        nodes = set(filtered_edges[0].tolist()) | set(filtered_edges[1].tolist())
        
        # Color nodes by labels if available
        if hasattr(self.data, 'y') and self.data.y is not None:
            colors = px.colors.qualitative.Set3
            for node in nodes:
                label = self.data.y[node].item() if node < len(self.data.y) else 0
                color = colors[label % len(colors)]
                net.add_node(node, label=str(node), color=color, 
                           title=f"Node {node}, Label: {label}")
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
        
        if save:
            filename = f"pyvis_network_layer_{layer_idx}_head_{head_idx}.html"
            filepath = os.path.join(self.save_dir, filename)
            net.save_graph(filepath)
            print(f"PyVis network saved to {filepath}")
        
        return net

    def generate_comprehensive_report(
        self,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive attention analysis report.

        Args:
            save: Whether to save the report.

        Returns:
            Dictionary with analysis results.
        """
        print("Generating comprehensive attention analysis report...")
        
        # Get attention analysis
        analysis = self.analyzer.analyze_attention_patterns(self.data)
        
        # Get graph statistics
        graph_stats = get_graph_statistics(self.data)
        
        # Create report
        report = {
            "graph_statistics": graph_stats,
            "attention_analysis": analysis,
            "model_info": self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
        }
        
        # Generate visualizations for each layer
        attention_data = self.analyzer.get_attention_weights(self.data)
        
        for layer_key in attention_data.keys():
            layer_idx = int(layer_key.split('_')[1])
            layer_data = attention_data[layer_key]
            
            print(f"Generating visualizations for {layer_key}...")
            
            # Attention distribution
            self.plot_attention_distribution(layer_idx=layer_idx, save=save)
            
            # Attention heatmap (first head)
            if layer_data["num_heads"] > 0:
                self.plot_attention_heatmap(layer_idx=layer_idx, head_idx=0, save=save)
            
            # Interactive plots
            self.create_interactive_attention_plot(layer_idx=layer_idx, head_idx=0, save=save)
            self.create_pyvis_network(layer_idx=layer_idx, head_idx=0, save=save)
        
        if save:
            import json
            filename = "attention_analysis_report.json"
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Analysis report saved to {filepath}")
        
        return report
