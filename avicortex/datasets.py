"""Brain graph connectivity datasets in torch Dataset class."""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data as PygData

from avicortex.builders import (
    CandiShareGraphBuilder,
    GraphBuilder,
    HCPGraphBuilder,
    OpenNeuroGraphBuilder,
)

ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, "data")


class GraphDataset(Dataset):
    """
    Base class for common functionalities of all datasets.

    Examples
    --------
    Data loading with batches:

    >>> from torch_geometric.loader import DenseDataLoader as PygDataLoader
    >>> tr_dataset = GraphDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=5)
    >>> for g in tr_dataloader:
    ...     print(g)

    Data loading without batching (no batch dimension):

    >>> from torch_geometric.loader import DataLoader as PygDataLoader
    >>> tr_dataset = GraphDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    >>> for g in tr_dataloader:
    ...     print(g)

    Data loading with batches but a view selection, useful if the task is graph-to-graph prediction:

    >>> from torch.utils.data import DataLoader
    >>> tr_dataset = GraphDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = DataLoader(tr_dataset, batch_size=5)
    >>> for g1, g2 in tr_dataloader:
    ...     print(g1)

    """

    def __init__(
        self,
        hemisphere: str,
        gbuilder: GraphBuilder,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        in_view_idx: int | None = None,
        out_view_idx: int | None = None,
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ):
        super().__init__()
        if hemisphere not in {"left", "right"}:
            raise ValueError("Hemisphere should be 'left' or 'right'.")
        self.mode = mode
        self.hemisphere = hemisphere
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.random_seed = random_seed

        self.n_folds = n_folds
        self.in_view_idx = in_view_idx
        self.out_view_idx = out_view_idx
        self.current_fold = current_fold
        self.gbuilder = gbuilder
        self.subjects_labels = self.gbuilder.get_labels()
        (self.subjects_nodes, self.subjects_edges) = self.gbuilder.construct(
            hem=self.hemisphere
        )
        self.subjects_nodes = self.subjects_nodes.transpose((1, 0, 2))
        self.subjects_edges = self.subjects_edges.transpose((2, 0, 1, 3))

        # Keep half of the data as 'unseen' to be used in inference.
        self.seen_data_indices, self.unseen_data_indices = self.get_fold_indices(
            self.subjects_labels.shape[0], 2, 0
        )

        if mode in {"train", "validation"}:
            self.seen_subjects_labels = self.subjects_labels[self.seen_data_indices]
            self.seen_subjects_nodes = self.subjects_nodes[self.seen_data_indices]
            self.seen_subjects_edges = self.subjects_edges[self.seen_data_indices]

            self.tr_indices, self.val_indices = self.get_fold_indices(
                self.seen_subjects_labels.shape[0],
                self.n_folds,
                self.current_fold,
            )

        if mode == "train":
            self.subjects_labels = self.seen_subjects_labels[self.tr_indices]
            self.subjects_nodes = self.seen_subjects_nodes[self.tr_indices]
            self.subjects_edges = self.seen_subjects_edges[self.tr_indices]
        elif mode == "validation":
            self.subjects_labels = self.seen_subjects_labels[self.val_indices]
            self.subjects_nodes = self.seen_subjects_nodes[self.val_indices]
            self.subjects_edges = self.seen_subjects_edges[self.val_indices]
        elif mode == "test":
            self.subjects_labels = self.subjects_labels[self.unseen_data_indices]
            self.subjects_nodes = self.subjects_nodes[self.unseen_data_indices]
            self.subjects_edges = self.subjects_edges[self.unseen_data_indices]
        elif mode == "inference":
            pass
        else:
            raise ValueError(
                "mode should be 'train', 'validation', 'test' or 'inference'"
            )

        self.n_subj, self.n_nodes, self.n_views = self.subjects_nodes.shape

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return source-target pair of the subject from a given index."""
        graph = self.get_view_graph_for_subject(index)
        in_view, out_view = graph, graph
        if self.in_view_idx is not None:
            in_view = self._select_view(graph, self.in_view_idx)
        if self.out_view_idx is not None:
            out_view = self._select_view(graph, self.out_view_idx)
        return in_view, out_view

    def __len__(self) -> int:
        """Return length of the dataset."""
        return self.n_subj

    @staticmethod
    def _select_view(graph: PygData, view_idx: int) -> PygData:
        """
        Select a single view from a given multigraph.

        Should be used only when view_idx was specified. Keeps view dimension by default.

        Parameters
        ----------
        graph: torch_geometric Data object (PygData)
            A multigraph to select a view from.

        Returns
        -------
        torch_geometric Data object (PygData)
            Selected view as a simple graph.
        """
        x = graph.x[:, :, view_idx : view_idx + 1] if graph.x is not None else None
        con_mat = (
            graph.con_mat[:, :, :, view_idx : view_idx + 1]
            if graph.con_mat is not None
            else None
        )
        edge_attr = (
            graph.edge_attr[:, :, view_idx : view_idx + 1]
            if graph.edge_attr is not None
            else None
        )

        return PygData(
            x=x,
            edge_index=graph.edge_index,
            edge_attr=edge_attr,
            con_mat=con_mat,
            y=graph.y,
        )

    def get_view_graph_for_subject(self, subj_idx: int) -> PygData:
        """
        For a single subject of a given index, combine data from different views and construct a multigraph.

        Parameters
        ----------
        subj_idx: int
            Index of the desired subject to construct its multigraph.

        Returns
        -------
        view_graph: torch_geometric Data object (PygData)
            A multigraph that encodes the brain of the subject.
        """
        view_graph = self.create_graph_obj(
            self.subjects_edges[subj_idx],
            self.subjects_nodes[subj_idx],
            self.subjects_labels[subj_idx : subj_idx + 1],
        )
        return view_graph

    def get_fold_indices(
        self, all_data_size: int, n_folds: int, fold_id: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create folds and get indices of train and validation datasets.

        Parameters
        ----------
        all_data_size: int
            Size of all data.
        fold_id: int
            Which cross validation fold to get the indices for.

        Returns
        -------
        train_indices: numpy ndarray
            Indices to get the training dataset.
        val_indices: numpy ndarray
            Indices to get the validation dataset.
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        split_indices = kf.split(range(all_data_size))
        train_indices, val_indices = [
            (np.array(train), np.array(val)) for train, val in split_indices
        ][fold_id]
        # Split train and test
        return train_indices, val_indices

    # Utility function to create a single multigraph from given numpy tensor: (n_rois, n_rois, n_views)
    def create_graph_obj(
        self, adj_matrix: np.ndarray, node_features: np.ndarray, labels: np.ndarray
    ) -> PygData:
        """
        Combine edges, nodes and labels to create a graph object for torch_geometric.

        Parameters
        ----------
        adj_matrix: numpy ndarray
            Adjacency matrix for the graph. Shaped (n_nodes, n_nodes, n_views)
        node_features: numpy ndarray
            Node feature matrix for the graph. Shaped (n_nodes, n_views)
        labels: numpy ndarray
            Subject-level labels. Only one label if batch size is 1.

        Returns
        -------
        torch_geometric Data object (PygData)
            Graph object designed to be used in the torch_geometric functions.
        """
        # Edge weights, ensure shape.
        edges = adj_matrix.reshape((self.n_nodes * self.n_nodes, self.n_views))
        # Torch operations execute faster to create source-destination pairs.
        # [0,1,2,3,0,1,2,3...]
        dst_index = torch.arange(self.n_nodes).repeat(self.n_nodes)
        # [0,0,0,0,1,1,1,1...]
        src_index = (
            torch.arange(self.n_nodes)
            .expand(self.n_nodes, self.n_nodes)
            .transpose(0, 1)
            .reshape(self.n_nodes * self.n_nodes)
        )
        # COO Matrix for index src-dst pairs. And add batch dimensions.
        edge_index = torch.stack([src_index, dst_index]).to(self.device)

        edge_attr = torch.from_numpy(edges).float().to(self.device).unsqueeze(0)
        x = torch.from_numpy(node_features).float().to(self.device).unsqueeze(0)
        y = torch.from_numpy(labels).float().to(self.device).unsqueeze(0)
        con_mat = torch.from_numpy(adj_matrix).float().to(self.device).unsqueeze(0)
        return PygData(
            x=x, edge_index=edge_index, edge_attr=edge_attr, con_mat=con_mat, y=y
        )

    def __repr__(self) -> str:
        """Dunder function to return string representation of the dataset."""
        return (
            f"{self.__class__.__name__} multigraph dataset ({self.mode}) {self.hemisphere} hemisphere with"
            f" n_views={self.n_views}, n_nodes={self.n_nodes}, n_subjects={self.n_subj}, "
            f"current fold:{self.current_fold + 1}/{self.n_folds}"
        )


class HCPYoungAdultDataset(GraphDataset):
    """
    Class to handle HCP Young Adult Dataset specificities.

    HCP Young Adult dataset:
    - 2 classes (male / female)
    - 1113 subjects
    - 34 nodes
    - 4 views

    Examples
    --------
    Data loading with batches:

    >>> from torch_geometric.loader import DenseDataLoader as PygDataLoader
    >>> tr_dataset = HCPYoungAdultDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=5)
    >>> for g in tr_dataloader:
    ...     print(g)

    Data loading without batching (no batch dimension):

    >>> from torch_geometric.loader import DataLoader as PygDataLoader
    >>> tr_dataset = HCPYoungAdultDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    >>> for g in tr_dataloader:
    ...     print(g)

    """

    def __init__(
        self,
        hemisphere: str,
        freesurfer_out_path: str | None = None,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        in_view_idx: int | None = None,
        out_view_idx: int | None = None,
    ):
        if freesurfer_out_path is None:
            freesurfer_out_path = os.path.join(DATA_PATH, "hcp_young_adult.csv")
        super().__init__(
            hemisphere,
            HCPGraphBuilder(freesurfer_out_path),
            mode,
            n_folds,
            current_fold,
            in_view_idx,
            out_view_idx,
        )


class OpenNeuroCannabisUsersDataset(GraphDataset):
    """
    Class to handle Openneuro cannabis users dataset specificities.

    This dataset has 2 scans per subject, at baseline and a 3 years follow up:
    - 2 classes (not user / cannabis user)
    - 42 subjects
    - 31 nodes
    - 4 views
    - 2 timepoints

    Examples
    --------
    Data loading with batches:

    >>> from torch_geometric.loader import DenseDataLoader as PygDataLoader
    >>> tr_dataset = OpenNeuroCannabisUsersDataset(hemisphere="left", mode="train", timepoint="baseline")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=5)
    >>> for g in tr_dataloader:
    ...     print(g)

    Data loading without batching (no batch dimension):

    >>> from torch_geometric.loader import DataLoader as PygDataLoader
    >>> tr_dataset = OpenNeuroCannabisUsersDataset(hemisphere="left", mode="train", timepoint="followup")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    >>> for g in tr_dataloader:
    ...     print(g)

    """

    def __init__(
        self,
        hemisphere: str,
        freesurfer_out_path: str | None = None,
        mode: str = "inference",
        timepoint: str | None = None,
        n_folds: int = 5,
        current_fold: int = 0,
        in_view_idx: int | None = None,
        out_view_idx: int | None = None,
    ):
        if freesurfer_out_path is None:
            if timepoint is None:
                freesurfer_out_path = os.path.join(
                    DATA_PATH, "openneuro_all_dktatlas.csv"
                )
            elif timepoint == "baseline":
                freesurfer_out_path = os.path.join(
                    DATA_PATH, "openneuro_baseline_dktatlas.csv"
                )
            elif timepoint == "followup":
                freesurfer_out_path = os.path.join(
                    DATA_PATH, "openneuro_followup_dktatlas.csv"
                )
            else:
                raise ValueError(
                    "timepoint should be one of: 'baseline', 'followup' or None."
                )
        include_all = freesurfer_out_path.endswith("openneuro_all_dktatlas.csv")
        super().__init__(
            hemisphere,
            OpenNeuroGraphBuilder(freesurfer_out_path, include_all),
            mode,
            n_folds,
            current_fold,
            in_view_idx,
            out_view_idx,
        )


class CandiShareSchizophreniaDataset(GraphDataset):
    """
    Class to handle Candi Share Schizophrenia Bulletin 2008 Dataset specificities.

    This dataset includes:
    - 4 classes (healthy / bipolar without psychosis / bipolar with psychosis / schizophrenia)
    - 103 subjects
    - 31 nodes
    - 4 views

    Examples
    --------
    Data loading with batches:

    >>> from torch_geometric.loader import DenseDataLoader as PygDataLoader
    >>> tr_dataset = CandiShareSchizophreniaDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=5)
    >>> for g in tr_dataloader:
    ...     print(g)

    Data loading without batching (no batch dimension):

    >>> from torch_geometric.loader import DataLoader as PygDataLoader
    >>> tr_dataset = CandiShareSchizophreniaDataset(hemisphere="left", mode="train")
    >>> tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    >>> for g in tr_dataloader:
    ...     print(g)

    """

    def __init__(
        self,
        hemisphere: str,
        freesurfer_out_path: str | None = None,
        mode: str = "inference",
        n_folds: int = 5,
        current_fold: int = 0,
        in_view_idx: int | None = None,
        out_view_idx: int | None = None,
    ):
        if freesurfer_out_path is None:
            freesurfer_out_path = os.path.join(
                DATA_PATH, "candishare_schizophrenia_dktatlas.csv"
            )
        super().__init__(
            hemisphere,
            CandiShareGraphBuilder(freesurfer_out_path),
            mode,
            n_folds,
            current_fold,
            in_view_idx,
            out_view_idx,
        )


# class ADNIAlzheimersDataset(GraphDataset):
#     """
#     ADNI Alzheimers Disease / Late Mild Cognitive Impairment Dataset:
#     - 2 classes (ad / lmci)
#     - 67 subjects
#     - 35 nodes
#     - 4 views
#     - 2 timepoints
#     """

#     def __init__(self):
#         super().__init__()

#     def __getitem__(self, index):
#         return

#     def __len__(self):
#         return
