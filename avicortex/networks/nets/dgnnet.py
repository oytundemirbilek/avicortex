"""DGN baseline model to predict a graph."""

from torch.nn import Linear, Sequential, Sigmoid
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData

from avicortex.networks.blocks.dgnblock import DGNBackbone
from avicortex.networks.utils import construct_graph_from_nodes


class DGNGraphPredictor(DGNBackbone):
    """A DGN to predict a graph from another graph."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv_size: int,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            conv_size,
        )

    def forward(self, graph: PygData) -> PygData:
        """Forward pass of the model."""
        learned_xs = super().forward(graph)
        # iterate over batch here.
        graph_list_for_batch = []
        for idx, x in enumerate(learned_xs):
            sample = graph.get_example(idx)
            reconstructed_graph = construct_graph_from_nodes(
                x, sample.edge_index, sample.y.squeeze(-1)
            )
            graph_list_for_batch.append(reconstructed_graph)
        # Return reconstructed graph at the last step, this should be the final generated graph.
        graph_batch = PygBatch()
        return graph_batch.from_data_list(graph_list_for_batch)


class DGNGraphClassifier(DGNBackbone):
    """A DGN to predict a class of a graph."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conv_size: int,
        n_classes: int,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            conv_size,
        )
        self.fc = Sequential(Linear(out_features, n_classes), Sigmoid())

    def forward(self, graph: PygData) -> PygData:
        """Forward pass of the model."""
        learned_xs = super().forward(graph)
        return self.fc(learned_xs)
