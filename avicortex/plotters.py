"""Module for graph plotting utility functions."""

from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch_geometric.data import Data as PygData


class Plotter:
    """Base class for common plotting utilities."""

    def __init__(self, figsize: Union[tuple[int, int], int, None] = None) -> None:
        self.figsize = figsize

    def create_subplots(self, n_plots: int) -> tuple[Figure, Axes]:
        """Create a number of subplots based on the figure size."""
        if self.figsize is None:
            fig, axes = plt.subplots(1, n_plots)
        elif isinstance(self.figsize, int):
            fig, axes = plt.subplots(
                1, n_plots, figsize=(self.figsize, self.figsize * n_plots)
            )
        else:
            fig, axes = plt.subplots(1, n_plots, figsize=self.figsize)

        return fig, axes


class GraphPlotter(Plotter):
    """Class for common plotting utilities for all graphs."""

    def __init__(self) -> None:
        super().__init__()

    def get_adjacency(self, graph: PygData) -> np.ndarray:
        return graph.edge_attr[batch, :, view].reshape(34, 34).cpu().detach().numpy()

    def plot_graph(self, graph: PygData, ax: Axes, title: str) -> None:
        g_adj = self.get_adjacency(graph)
        ax.matshow(g_adj)
        ax.title.set_text(title)
        ax.set_axis_off()

    def plot_graph_list(
        self,
        graph_list: list[PygData],
        titles: list[Any],
        batch: int = 0,
        view: int = 0,
    ) -> None:
        """Plot adjacency matrices of the given graphs in a list."""
        n_plots = len(graph_list)
        _, axes = self.create_subplots(n_plots)
        for g_idx, graph in enumerate(graph_list):
            self.plot_graph(graph, axes[g_idx], titles[g_idx])

        plt.tight_layout()
        plt.show()
