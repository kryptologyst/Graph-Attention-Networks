"""Graph Attention Networks (GAT) implementation with modern features."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv.gat_conv import GATConv as PyGGATConv
from torch_geometric.typing import Adj, OptTensor, PairTensor


class GATLayer(nn.Module):
    """Single Graph Attention Layer with enhanced features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        flow: str = "source_to_target",
        node_dim: int = 0,
        attention_type: str = "standard",
    ) -> None:
        """Initialize GAT layer.

        Args:
            in_channels: Size of each input sample.
            out_channels: Size of each output sample.
            heads: Number of multi-head-attentions.
            concat: If set to False, multi-head attentions are averaged instead of concatenated.
            negative_slope: LeakyReLU angle of the negative slope.
            dropout: Dropout probability of the normalized attention coefficients.
            add_self_loops: If set to True, will add self-loops to the input graph.
            bias: If set to False, the layer will not learn an additive bias.
            share_weights: If set to True, the same matrix will be applied to the source and target node.
            edge_dim: Edge feature dimensionality.
            fill_value: The way to generate edge features of self-loops.
            flow: The flow direction when using in combination with message passing.
            node_dim: The axis along which to propagate.
            attention_type: Type of attention mechanism ('standard', 'sparse', 'dynamic').
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.bias = bias
        self.share_weights = share_weights
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.flow = flow
        self.node_dim = node_dim
        self.attention_type = attention_type

        # Initialize the GAT convolution layer
        self.gat_conv = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias,
            share_weights=share_weights,
            edge_dim=edge_dim,
            fill_value=fill_value,
            flow=flow,
            node_dim=node_dim,
        )

        # Additional components for enhanced GAT
        self.layer_norm = nn.LayerNorm(out_channels * heads if concat else out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels * heads if concat else out_channels)
        self.use_layer_norm = False
        self.use_batch_norm = False

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Optional[Tuple[int, int]] = None,
        return_attention_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        """Forward pass of the GAT layer.

        Args:
            x: Node feature matrix of shape [N, in_channels] or edge features.
            edge_index: Graph connectivity in COO format with shape [2, M].
            edge_attr: Edge feature matrix with shape [M, edge_dim].
            size: The size [N, M] of the assignment matrix in bipartite graphs.
            return_attention_weights: If set to True, will return the attention weights.

        Returns:
            Node embeddings of shape [N, out_channels] or [N, heads * out_channels].
            Optionally, attention weights if return_attention_weights is True.
        """
        # Apply GAT convolution
        if return_attention_weights:
            out, attention_weights = self.gat_conv(
                x, edge_index, edge_attr, size, return_attention_weights=True
            )
        else:
            out = self.gat_conv(x, edge_index, edge_attr, size)

        # Apply normalization
        if self.use_layer_norm:
            out = self.layer_norm(out)
        elif self.use_batch_norm:
            out = self.batch_norm(out)

        if return_attention_weights:
            return out, attention_weights
        return out

    def enable_layer_norm(self) -> None:
        """Enable layer normalization."""
        self.use_layer_norm = True
        self.use_batch_norm = False

    def enable_batch_norm(self) -> None:
        """Enable batch normalization."""
        self.use_batch_norm = True
        self.use_layer_norm = False


class GAT(nn.Module):
    """Graph Attention Network with modern architecture and features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.6,
        alpha: float = 0.2,
        bias: bool = True,
        concat: bool = True,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        flow: str = "source_to_target",
        node_dim: int = 0,
        share_weights: bool = False,
        attention_type: str = "standard",
        attention_dropout: float = 0.6,
        attention_activation: str = "elu",
        final_activation: str = "log_softmax",
        use_layer_norm: bool = False,
        use_batch_norm: bool = False,
        residual_connections: bool = False,
    ) -> None:
        """Initialize GAT model.

        Args:
            in_channels: Size of each input sample.
            hidden_channels: Size of each hidden sample.
            out_channels: Size of each output sample.
            num_layers: Number of GAT layers.
            heads: Number of multi-head-attentions.
            dropout: Dropout probability.
            alpha: LeakyReLU angle of the negative slope.
            bias: If set to False, the layer will not learn an additive bias.
            concat: If set to False, multi-head attentions are averaged instead of concatenated.
            add_self_loops: If set to True, will add self-loops to the input graph.
            edge_dim: Edge feature dimensionality.
            fill_value: The way to generate edge features of self-loops.
            flow: The flow direction when using in combination with message passing.
            node_dim: The axis along which to propagate.
            share_weights: If set to True, the same matrix will be applied to the source and target node.
            attention_type: Type of attention mechanism.
            attention_dropout: Dropout probability of the normalized attention coefficients.
            attention_activation: Activation function for attention layers.
            final_activation: Activation function for the final layer.
            use_layer_norm: Whether to use layer normalization.
            use_batch_norm: Whether to use batch normalization.
            residual_connections: Whether to use residual connections.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.bias = bias
        self.concat = concat
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.flow = flow
        self.node_dim = node_dim
        self.share_weights = share_weights
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout
        self.attention_activation = attention_activation
        self.final_activation = final_activation
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.residual_connections = residual_connections

        # Build GAT layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GATLayer(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                concat=concat,
                negative_slope=alpha,
                dropout=attention_dropout,
                add_self_loops=add_self_loops,
                bias=bias,
                share_weights=share_weights,
                edge_dim=edge_dim,
                fill_value=fill_value,
                flow=flow,
                node_dim=node_dim,
                attention_type=attention_type,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATLayer(
                    in_channels=hidden_channels * heads if concat else hidden_channels,
                    out_channels=hidden_channels,
                    heads=heads,
                    concat=concat,
                    negative_slope=alpha,
                    dropout=attention_dropout,
                    add_self_loops=add_self_loops,
                    bias=bias,
                    share_weights=share_weights,
                    edge_dim=edge_dim,
                    fill_value=fill_value,
                    flow=flow,
                    node_dim=node_dim,
                    attention_type=attention_type,
                )
            )

        # Final layer
        if num_layers > 1:
            self.layers.append(
                GATLayer(
                    in_channels=hidden_channels * heads if concat else hidden_channels,
                    out_channels=out_channels,
                    heads=1,
                    concat=False,
                    negative_slope=alpha,
                    dropout=attention_dropout,
                    add_self_loops=add_self_loops,
                    bias=bias,
                    share_weights=share_weights,
                    edge_dim=edge_dim,
                    fill_value=fill_value,
                    flow=flow,
                    node_dim=node_dim,
                    attention_type=attention_type,
                )
            )

        # Enable normalization if specified
        if use_layer_norm:
            for layer in self.layers:
                layer.enable_layer_norm()
        elif use_batch_norm:
            for layer in self.layers:
                layer.enable_batch_norm()

        # Input dropout
        self.input_dropout = nn.Dropout(dropout)

        # Activation functions
        self.activation = self._get_activation(attention_activation)
        self.final_activation_fn = self._get_activation(final_activation)

    def _get_activation(self, activation: str) -> callable:
        """Get activation function by name."""
        activations = {
            "relu": F.relu,
            "elu": F.elu,
            "leaky_relu": F.leaky_relu,
            "gelu": F.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax,
            "log_softmax": F.log_softmax,
            "none": lambda x: x,
        }
        return activations.get(activation, F.elu)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = False,
        return_embeddings: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tuple[Tensor, Tensor]]], Tuple[Tensor, Tensor]]:
        """Forward pass of the GAT model.

        Args:
            x: Node feature matrix of shape [N, in_channels].
            edge_index: Graph connectivity in COO format with shape [2, M].
            edge_attr: Edge feature matrix with shape [M, edge_dim].
            return_attention_weights: If set to True, will return attention weights for all layers.
            return_embeddings: If set to True, will return intermediate embeddings.

        Returns:
            Node embeddings or logits of shape [N, out_channels].
            Optionally, attention weights and/or intermediate embeddings.
        """
        # Apply input dropout
        x = self.input_dropout(x)

        # Store attention weights and embeddings if requested
        attention_weights = []
        embeddings = [x]

        # Forward pass through layers
        for i, layer in enumerate(self.layers):
            # Store previous layer output for residual connection
            prev_x = x if self.residual_connections else None

            # Apply layer
            if return_attention_weights:
                x, attn_weights = layer(
                    x, edge_index, edge_attr, return_attention_weights=True
                )
                attention_weights.append(attn_weights)
            else:
                x = layer(x, edge_index, edge_attr)

            # Apply activation (except for the last layer)
            if i < len(self.layers) - 1:
                x = self.activation(x)

            # Apply dropout
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection
            if self.residual_connections and prev_x is not None:
                # Ensure dimensions match for residual connection
                if x.shape == prev_x.shape:
                    x = x + prev_x
                elif x.shape[1] == prev_x.shape[1]:
                    x = x + prev_x

            # Store embeddings
            if return_embeddings:
                embeddings.append(x)

        # Apply final activation
        if self.final_activation != "none":
            x = self.final_activation_fn(x, dim=1)

        # Prepare return values
        if return_attention_weights and return_embeddings:
            return x, attention_weights, embeddings
        elif return_attention_weights:
            return x, attention_weights
        elif return_embeddings:
            return x, embeddings
        else:
            return x

    def get_attention_weights(
        self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> List[Tuple[Tensor, Tensor]]:
        """Get attention weights for all layers.

        Args:
            x: Node feature matrix.
            edge_index: Graph connectivity.
            edge_attr: Edge features.

        Returns:
            List of attention weight tuples for each layer.
        """
        _, attention_weights = self.forward(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        return attention_weights

    def get_embeddings(
        self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None
    ) -> List[Tensor]:
        """Get intermediate embeddings from all layers.

        Args:
            x: Node feature matrix.
            edge_index: Graph connectivity.
            edge_attr: Edge features.

        Returns:
            List of embeddings from each layer.
        """
        _, embeddings = self.forward(x, edge_index, edge_attr, return_embeddings=True)
        return embeddings

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics.

        Returns:
            Dictionary containing model information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "GAT",
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "num_layers": self.num_layers,
            "heads": self.heads,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "attention_type": self.attention_type,
            "residual_connections": self.residual_connections,
            "use_layer_norm": self.use_layer_norm,
            "use_batch_norm": self.use_batch_norm,
        }
