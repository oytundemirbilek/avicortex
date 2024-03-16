"""Brain graph connectivity datasets in torch Dataset class."""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data as PygData

from avicortex.builders import (
    ADNIGraphBuilder,
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

    def __init__(  # noqa: PLR0915, PLR0913, PLR0917
        self,
        hemisphere: str,
        mode: str = "inference",
        n_folds: int | None = None,
        current_fold: int = 0,
        data_split_ratio: tuple[int, int, int] = (4, 1, 5),
        src_view_idx: int | None = None,
        tgt_view_idx: int | None = None,
        src_atlas: str = "dktatlas",
        tgt_atlas: str = "dktatlas",
        src_atlas_path: str | None = None,
        tgt_atlas_path: str | None = None,
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ):
        super().__init__()
        if n_folds is not None:
            warnings.warn(
                "n_folds is deprecated, use train-validation-test splits with data_split_ratio=(4,1,5) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if hemisphere not in {"left", "right"}:
            raise ValueError("Hemisphere should be 'left' or 'right'.")
        self.mode = mode
        self.hemisphere = hemisphere
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.random_seed = random_seed
        if data_split_ratio[1] > 1:
            raise ValueError("Validation split should be 1. Define others based on it.")

        self.n_folds = data_split_ratio[0] + data_split_ratio[1]
        self.src_view_idx = src_view_idx
        self.tgt_view_idx = tgt_view_idx
        self.src_atlas_path = src_atlas_path
        self.tgt_atlas_path = tgt_atlas_path
        self.src_atlas = src_atlas
        self.tgt_atlas = tgt_atlas
        self.current_fold = current_fold
        self.data_split_ratio = data_split_ratio
        self.gbuilder = self.get_graph_builder()
        self.subjects_nodes_src, self.subjects_edges_src = np.array([]), np.array([])
        self.subjects_nodes_tgt, self.subjects_edges_tgt = np.array([]), np.array([])
        self.init_load_atlases()
        self.subjects_ids = self.gbuilder.get_subject_ids().to_numpy()
        self.init_arrange_splits()
        self.subjects_labels = self.gbuilder.get_labels()

        if self.mode in {"train", "validation"}:
            self.seen_subjects_labels = self.subjects_labels[self.seen_indices]
            self.seen_subjects_nodes_src = self.subjects_nodes_src[self.seen_indices]
            self.seen_subjects_nodes_tgt = self.subjects_nodes_tgt[self.seen_indices]
            self.seen_subjects_edges_src = self.subjects_edges_src[self.seen_indices]
            self.seen_subjects_edges_tgt = self.subjects_edges_tgt[self.seen_indices]
            self.seen_subjects_ids = self.subjects_ids[self.seen_indices]

        if self.mode == "train":
            self.subjects_labels = self.seen_subjects_labels[self.tr_indices]
            self.subjects_nodes_src = self.seen_subjects_nodes_src[self.tr_indices]
            self.subjects_nodes_tgt = self.seen_subjects_nodes_tgt[self.tr_indices]
            self.subjects_edges_src = self.seen_subjects_edges_src[self.tr_indices]
            self.subjects_edges_tgt = self.seen_subjects_edges_tgt[self.tr_indices]
            self.subjects_ids = self.seen_subjects_ids[self.tr_indices]
        elif self.mode == "validation":
            self.subjects_labels = self.seen_subjects_labels[self.val_indices]
            self.subjects_nodes_src = self.seen_subjects_nodes_src[self.val_indices]
            self.subjects_nodes_tgt = self.seen_subjects_nodes_tgt[self.val_indices]
            self.subjects_edges_src = self.seen_subjects_edges_src[self.val_indices]
            self.subjects_edges_tgt = self.seen_subjects_edges_tgt[self.val_indices]
            self.subjects_ids = self.seen_subjects_ids[self.val_indices]
        elif self.mode == "test":
            self.subjects_labels = self.subjects_labels[self.unseen_indices]
            self.subjects_nodes_src = self.subjects_nodes_src[self.unseen_indices]
            self.subjects_nodes_tgt = self.subjects_nodes_tgt[self.unseen_indices]
            self.subjects_edges_src = self.subjects_edges_src[self.unseen_indices]
            self.subjects_edges_tgt = self.subjects_edges_tgt[self.unseen_indices]
            self.subjects_ids = self.subjects_ids[self.unseen_indices]
        elif self.mode == "inference":
            pass
        else:
            raise ValueError(
                "mode should be 'train', 'validation', 'test' or 'inference'"
            )

        self.n_subj, self.n_nodes_src, self.n_views_src = self.subjects_nodes_src.shape
        self.n_subj, self.n_nodes_tgt, self.n_views_tgt = self.subjects_nodes_tgt.shape

    @classmethod
    def get_graph_builder(cls) -> GraphBuilder:
        """Get graph builder specific to the dataset."""
        return GraphBuilder()

    def init_arrange_splits(self) -> None:
        """Assign indices to the train-validation-test splits."""
        if self.data_split_ratio[2] > 0:
            self.test_folds = (
                self.data_split_ratio[0]
                + self.data_split_ratio[1]
                + self.data_split_ratio[2]
            ) // self.data_split_ratio[2]
        else:
            self.test_folds = 0

        if self.test_folds > 1:
            # Keep partition of the data as 'unseen' to be used as test split.
            self.seen_indices, self.unseen_indices = self.get_fold_indices(
                self.subjects_ids.shape[0], self.test_folds, 0, self.random_seed
            )
        elif self.test_folds == 1:
            self.seen_indices = np.array([], dtype=np.int32)
            self.unseen_indices = np.arange(self.subjects_ids.shape[0], dtype=np.int32)
        elif self.test_folds == 0:
            self.unseen_indices = np.array([], dtype=np.int32)
            self.seen_indices = np.arange(self.subjects_ids.shape[0], dtype=np.int32)

        if self.mode in {"train", "validation"}:
            self.tr_indices, self.val_indices = self.get_fold_indices(
                self.seen_indices.shape[0],
                self.n_folds,
                self.current_fold,
                self.random_seed,
            )

    def init_load_atlases(self) -> None:
        """Load node and edge feature sets based on the source and target atlases."""
        # Source atlas
        if self.src_atlas_path is None:
            raise ValueError("Source atlas path is not provided.")
        if self.tgt_atlas_path is None:
            raise ValueError("Target atlas path is not provided.")
        self.gbuilder.load_atlas(
            fs_stats_atlas_path=self.src_atlas_path, atlas=self.src_atlas
        )
        (self.subjects_nodes_src, self.subjects_edges_src) = self.gbuilder.construct(
            hem=self.hemisphere
        )
        self.subjects_nodes_src = self.subjects_nodes_src.transpose((1, 0, 2))
        self.subjects_edges_src = self.subjects_edges_src.transpose((2, 0, 1, 3))

        # Target atlas
        self.gbuilder.load_atlas(
            fs_stats_atlas_path=self.tgt_atlas_path, atlas=self.tgt_atlas
        )
        (self.subjects_nodes_tgt, self.subjects_edges_tgt) = self.gbuilder.construct(
            hem=self.hemisphere
        )
        self.subjects_nodes_tgt = self.subjects_nodes_tgt.transpose((1, 0, 2))
        self.subjects_edges_tgt = self.subjects_edges_tgt.transpose((2, 0, 1, 3))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return source-target pair of the subject from a given index."""
        src_graph = self.get_view_graph_for_subject(index, tgt=False)
        tgt_graph = self.get_view_graph_for_subject(index, tgt=True)
        if self.src_view_idx is not None:
            src_graph = self._select_view(src_graph, self.src_view_idx)
        if self.tgt_view_idx is not None:
            tgt_graph = self._select_view(tgt_graph, self.tgt_view_idx)
        return src_graph, tgt_graph

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
        subject_id = graph.subject_id if graph.subject_id is not None else None
        y = graph.y if graph.y is not None else None

        return PygData(
            x=x,
            edge_index=graph.edge_index,
            edge_attr=edge_attr,
            con_mat=con_mat,
            y=y,
            subject_id=subject_id,
        )

    def get_view_graph_for_subject(self, subj_idx: int, tgt: bool = False) -> PygData:
        """
        For a single subject of a given index, combine data from different views and construct a multigraph.

        Parameters
        ----------
        subj_idx: int
            Index of the desired subject to construct its multigraph.
        tgt: boolean
            Whether the requested graph is a target or not.

        Returns
        -------
        view_graph: torch_geometric Data object (PygData)
            A multigraph that encodes the brain of the subject.
        """
        if tgt:
            return self.create_graph_obj(
                self.subjects_edges_tgt[subj_idx],
                self.subjects_nodes_tgt[subj_idx],
                self.subjects_labels[subj_idx : subj_idx + 1],
                self.subjects_ids[subj_idx],
                tgt,
            )
        else:
            return self.create_graph_obj(
                self.subjects_edges_src[subj_idx],
                self.subjects_nodes_src[subj_idx],
                self.subjects_labels[subj_idx : subj_idx + 1],
                self.subjects_ids[subj_idx],
                tgt,
            )

    @staticmethod
    def get_fold_indices(
        all_data_size: int, n_folds: int, fold_id: int = 0, random_seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create folds and get indices of train and validation datasets.

        Parameters
        ----------
        all_data_size: int
            Size of all data.
        fold_id: int
            Which cross validation fold to get the indices for.
        random_seed: int
            Random seed to be used for randomization.

        Returns
        -------
        train_indices: numpy ndarray
            Indices to get the training dataset.
        val_indices: numpy ndarray
            Indices to get the validation dataset.
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_indices = kf.split(range(all_data_size))
        train_indices, val_indices = [
            (np.array(train), np.array(val)) for train, val in split_indices
        ][fold_id]
        # Split train and test
        return train_indices, val_indices

    # Utility function to create a single multigraph from given numpy tensor: (n_rois, n_rois, n_views)
    def create_graph_obj(
        self,
        adj_matrix: np.ndarray,
        node_features: np.ndarray,
        labels: np.ndarray,
        subject_id: str,
        tgt: bool = True,
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
        subject_id: string
            Subject ID attached to the graph.
        tgt: boolean
            Whether the requested graph is a target or not.

        Returns
        -------
        torch_geometric Data object (PygData)
            Graph object designed to be used in the torch_geometric functions.
        """
        # Edge weights, ensure shape.
        n_nodes = self.n_nodes_src
        n_views = self.n_views_src
        if tgt:
            n_nodes = self.n_nodes_tgt
            n_views = self.n_views_tgt
        edges = adj_matrix.reshape((n_nodes * n_nodes, n_views))
        # Torch operations execute faster to create source-destination pairs.
        # [0,1,2,3,0,1,2,3...]
        dst_index = torch.arange(n_nodes).repeat(n_nodes)
        # [0,0,0,0,1,1,1,1...]
        src_index = (
            torch.arange(n_nodes)
            .expand(n_nodes, n_nodes)
            .transpose(0, 1)
            .reshape(n_nodes * n_nodes)
        )
        # COO Matrix for index src-dst pairs. And add batch dimensions.
        edge_index = torch.stack([src_index, dst_index]).to(self.device)

        edge_attr = torch.from_numpy(edges).float().to(self.device).unsqueeze(0)
        x = torch.from_numpy(node_features).float().to(self.device).unsqueeze(0)
        y = torch.from_numpy(labels).float().to(self.device).unsqueeze(0)
        con_mat = torch.from_numpy(adj_matrix).float().to(self.device).unsqueeze(0)
        return PygData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            con_mat=con_mat,
            y=y,
            subject_id=subject_id,
        )

    def __repr__(self) -> str:
        """Dunder function to return string representation of the dataset."""
        return (
            f"{self.__class__.__name__} multigraph dataset ({self.mode}) {self.hemisphere} hemisphere with"
            f", n_subjects={self.n_subj}, current fold:{self.current_fold + 1}/{self.n_folds}"
            f", n_views_source={self.n_views_src}, n_nodes_source={self.n_nodes_src}"
            f", n_views_target={self.n_views_tgt}, n_nodes_target={self.n_nodes_tgt}"
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

    def __init__(  # noqa: PLR0915, PLR0913, PLR0917
        self,
        hemisphere: str,
        freesurfer_out_path: str | None = None,
        mode: str = "inference",
        n_folds: int | None = None,
        current_fold: int = 0,
        data_split_ratio: tuple[int, int, int] = (4, 1, 5),
        src_view_idx: int | None = None,
        tgt_view_idx: int | None = None,
        src_atlas: str = "dktatlas",
        tgt_atlas: str = "dktatlas",
        src_atlas_path: str | None = None,
        tgt_atlas_path: str | None = None,
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ):
        default_file = "hcp_young_adult.csv"
        src_atlas_path_ref = src_atlas_path
        tgt_atlas_path_ref = tgt_atlas_path
        if src_atlas_path_ref is None:
            src_atlas_path_ref = os.path.join(DATA_PATH, default_file)
        if tgt_atlas_path_ref is None:
            tgt_atlas_path_ref = os.path.join(DATA_PATH, default_file)

        if freesurfer_out_path is not None:
            warnings.warn(
                "freesurfer_out_path is deprecated, use src_atlas_path and tgt_atlas_path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            src_atlas_path_ref = freesurfer_out_path
            tgt_atlas_path_ref = freesurfer_out_path
        super().__init__(
            hemisphere=hemisphere,
            mode=mode,
            n_folds=n_folds,
            current_fold=current_fold,
            data_split_ratio=data_split_ratio,
            src_view_idx=src_view_idx,
            tgt_view_idx=tgt_view_idx,
            src_atlas=src_atlas,
            tgt_atlas=tgt_atlas,
            src_atlas_path=src_atlas_path_ref,
            tgt_atlas_path=tgt_atlas_path_ref,
            device=device,
            random_seed=random_seed,
        )

    @classmethod
    def get_graph_builder(cls) -> GraphBuilder:
        """Get graph builder specific to the dataset."""
        return HCPGraphBuilder()


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

    def __init__(  # noqa: PLR0915, PLR0913, PLR0917
        self,
        hemisphere: str,
        freesurfer_out_path: str | None = None,
        mode: str = "inference",
        timepoint: str | None = None,
        n_folds: int | None = None,
        current_fold: int = 0,
        data_split_ratio: tuple[int, int, int] = (4, 1, 5),
        src_view_idx: int | None = None,
        tgt_view_idx: int | None = None,
        src_atlas: str = "dktatlas",
        tgt_atlas: str = "dktatlas",
        src_atlas_path: str | None = None,
        tgt_atlas_path: str | None = None,
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ):
        src_atlas_path_ref = src_atlas_path
        tgt_atlas_path_ref = tgt_atlas_path
        if timepoint is None:
            if src_atlas_path_ref is None:
                src_atlas_path_ref = os.path.join(
                    DATA_PATH, f"openneuro_baseline_{src_atlas}.csv"
                )
            if tgt_atlas_path_ref is None:
                tgt_atlas_path_ref = os.path.join(
                    DATA_PATH, f"openneuro_followup_{tgt_atlas}.csv"
                )
        else:
            if src_atlas_path_ref is None:
                src_atlas_path_ref = os.path.join(
                    DATA_PATH, f"openneuro_{timepoint}_{src_atlas}.csv"
                )
            if tgt_atlas_path_ref is None:
                tgt_atlas_path_ref = os.path.join(
                    DATA_PATH, f"openneuro_{timepoint}_{tgt_atlas}.csv"
                )
            else:
                warnings.warn(
                    "Data will be loaded from the provided paths, even if timepoint is set differently.",
                    UserWarning,
                    stacklevel=2,
                )
                src_atlas_path_ref = src_atlas_path
                tgt_atlas_path_ref = tgt_atlas_path

        if freesurfer_out_path is not None:
            warnings.warn(
                "freesurfer_out_path is deprecated, use src_atlas_path and tgt_atlas_path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            src_atlas_path_ref = freesurfer_out_path
            tgt_atlas_path_ref = freesurfer_out_path

        super().__init__(
            hemisphere=hemisphere,
            mode=mode,
            n_folds=n_folds,
            current_fold=current_fold,
            data_split_ratio=data_split_ratio,
            src_view_idx=src_view_idx,
            tgt_view_idx=tgt_view_idx,
            src_atlas=src_atlas,
            tgt_atlas=tgt_atlas,
            src_atlas_path=src_atlas_path_ref,
            tgt_atlas_path=tgt_atlas_path_ref,
            device=device,
            random_seed=random_seed,
        )

    @classmethod
    def get_graph_builder(cls) -> GraphBuilder:
        """Get graph builder specific to the dataset."""
        return OpenNeuroGraphBuilder()


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

    def __init__(  # noqa: PLR0915, PLR0913, PLR0917
        self,
        hemisphere: str,
        freesurfer_out_path: str | None = None,
        mode: str = "inference",
        n_folds: int | None = None,
        current_fold: int = 0,
        data_split_ratio: tuple[int, int, int] = (4, 1, 5),
        src_view_idx: int | None = None,
        tgt_view_idx: int | None = None,
        src_atlas: str = "dktatlas",
        tgt_atlas: str = "dktatlas",
        src_atlas_path: str | None = None,
        tgt_atlas_path: str | None = None,
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ):
        default_file = "candishare_schizophrenia_dktatlas.csv"
        src_atlas_path_ref = src_atlas_path
        tgt_atlas_path_ref = tgt_atlas_path
        if src_atlas_path_ref is None:
            src_atlas_path_ref = os.path.join(DATA_PATH, default_file)
        if tgt_atlas_path_ref is None:
            tgt_atlas_path_ref = os.path.join(DATA_PATH, default_file)
        if freesurfer_out_path is not None:
            warnings.warn(
                "freesurfer_out_path is deprecated, use src_atlas_path and tgt_atlas_path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            src_atlas_path_ref = freesurfer_out_path
            tgt_atlas_path_ref = freesurfer_out_path
        super().__init__(
            hemisphere=hemisphere,
            mode=mode,
            n_folds=n_folds,
            current_fold=current_fold,
            data_split_ratio=data_split_ratio,
            src_view_idx=src_view_idx,
            tgt_view_idx=tgt_view_idx,
            src_atlas=src_atlas,
            tgt_atlas=tgt_atlas,
            src_atlas_path=src_atlas_path_ref,
            tgt_atlas_path=tgt_atlas_path_ref,
            device=device,
            random_seed=random_seed,
        )

    @classmethod
    def get_graph_builder(cls) -> GraphBuilder:
        """Get graph builder specific to the dataset."""
        return CandiShareGraphBuilder()


class ADNIAlzheimersDataset(GraphDataset):
    """
    Class to handle Alzheimers Disease Neuroimaging Initiative (ADNI) dataset specificities.

    This dataset has 2 scans per subject:
    - 2 classes (not user / cannabis user)
    - ... subjects
    - 34 nodes
    - 4 views

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

    def __init__(  # noqa: PLR0915, PLR0913, PLR0917
        self,
        hemisphere: str,
        freesurfer_out_path: str | None = None,
        mode: str = "inference",
        n_folds: int | None = None,
        current_fold: int = 0,
        data_split_ratio: tuple[int, int, int] = (4, 1, 5),
        src_view_idx: int | None = None,
        tgt_view_idx: int | None = None,
        src_atlas: str = "dktatlas",
        tgt_atlas: str = "dktatlas",
        src_atlas_path: str | None = None,
        tgt_atlas_path: str | None = None,
        device: str | torch.device | None = None,
        random_seed: int = 0,
    ) -> None:
        default_file = "adni3.csv"
        src_atlas_path_ref = src_atlas_path
        tgt_atlas_path_ref = tgt_atlas_path
        if src_atlas_path is None:
            src_atlas_path_ref = os.path.join(DATA_PATH, default_file)
        if tgt_atlas_path is None:
            tgt_atlas_path_ref = os.path.join(DATA_PATH, default_file)
        if freesurfer_out_path is not None:
            warnings.warn(
                "freesurfer_out_path is deprecated, use src_atlas_path and tgt_atlas_path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            src_atlas_path_ref = freesurfer_out_path
            tgt_atlas_path_ref = freesurfer_out_path

        super().__init__(
            hemisphere=hemisphere,
            mode=mode,
            n_folds=n_folds,
            current_fold=current_fold,
            data_split_ratio=data_split_ratio,
            src_view_idx=src_view_idx,
            tgt_view_idx=tgt_view_idx,
            src_atlas=src_atlas,
            tgt_atlas=tgt_atlas,
            src_atlas_path=src_atlas_path_ref,
            tgt_atlas_path=tgt_atlas_path_ref,
            device=device,
            random_seed=random_seed,
        )

    @classmethod
    def get_graph_builder(cls) -> GraphBuilder:
        """Get graph builder specific to the dataset."""
        return ADNIGraphBuilder(os.path.join(DATA_PATH, "adni3_region_mapping.csv"))
