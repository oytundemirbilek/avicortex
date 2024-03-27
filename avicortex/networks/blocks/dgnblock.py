"""Module for DGN blocks."""

from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential
from torch_geometric.data import Data as PygData
from torch_geometric.nn import NNConv
from torch_geometric.nn import Sequential as PygSequential


class DGNBlock(Module):
    """A single GNN layer in a DGN."""

    def __init__(self, n_features: int, conv_size: int, aggr: str = "mean") -> None:
        super().__init__()
        self.nn = Sequential(Linear(n_features, n_features * conv_size), ReLU())
        self.conv1 = NNConv(n_features, conv_size, self.nn, aggr=aggr)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method of the block."""
        return self.conv1(x)


class DGNBackbone(Module):
    """GNN layers that a DGN consists of."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv_size: int,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.conv_size = conv_size

        # input-output definitions for PygSequential parameters
        g_in = "x, edge_index, edge_attr"
        g_in_out = "x, edge_index, edge_attr -> x"

        self.conv1 = DGNBlock(self.in_features, self.conv_size, aggr="mean")
        self.conv2 = DGNBlock(self.conv_size, self.conv_size, aggr="mean")
        self.conv3 = DGNBlock(self.conv_size, self.out_features, aggr="mean")

        self.nnconv_layers = PygSequential(
            g_in,
            [(self.conv1, g_in_out), (self.conv2, g_in_out), (self.conv3, g_in_out)],
        )

    def forward(self, graph: PygData) -> PygData:
        """Forward pass of the model."""
        learned_xs = self.nnconv_layers(
            x=graph.x.flatten(end_dim=-2),
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr.flatten(end_dim=-2),
        )
        return learned_xs.reshape_as(graph.x)
