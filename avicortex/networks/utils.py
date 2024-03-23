"""Utility functions for handling graph data."""

import torch
from torch import Tensor
from torch_geometric.data import Data as PygData


def construct_graph_from_nodes(nodes: Tensor, edge_index: Tensor, y: Tensor) -> PygData:
    """Construct edges by taking pairwise absolute difference between node features."""
    reconstructed_edges = (torch.abs(nodes.unsqueeze(1) - nodes)).flatten(end_dim=-2)

    reconstructed_graph = PygData(
        x=nodes.unsqueeze(0),
        edge_index=edge_index,
        edge_attr=reconstructed_edges.unsqueeze(0),
        y=y.unsqueeze(0),
    )
    return reconstructed_graph
