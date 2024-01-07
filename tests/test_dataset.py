"""Test graph dataset classes."""
import torch
from torch_geometric.loader import DataLoader as PygDataLoader

from avicortex.datasets import OpenNeuroCannabisUsersDataset


def test_simple_iteration() -> None:
    """Test if a dataset can be iterated."""
    n_views = 4
    n_nodes = 34
    dataset_obj = OpenNeuroCannabisUsersDataset(hemisphere="left", timepoint="baseline")
    dataloader = PygDataLoader(dataset_obj, batch_size=1)
    assert dataset_obj.n_nodes == n_nodes
    assert dataset_obj.n_views == n_views
    src_graph, tgt_graph = next(iter(dataloader))
    assert src_graph.x is not None
    assert src_graph.edge_index is not None
    assert src_graph.edge_attr is not None
    assert src_graph.con_mat is not None
    assert src_graph.y is not None
    assert src_graph.x.shape == (1, n_nodes, n_views)
    assert src_graph.edge_index.shape == (2, n_nodes * n_nodes)
    assert src_graph.edge_attr.shape == (1, n_nodes * n_nodes, n_views)
    assert src_graph.con_mat.shape == (1, n_nodes, n_nodes, n_views)
    assert src_graph.y.shape == (1, 1)

    assert torch.equal(src_graph.y, tgt_graph.y)


def test_hemispheres() -> None:
    """Test if the dataset can read different hemispheres correctly."""
    left_dataset_obj = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline"
    )
    left_dataloader = PygDataLoader(left_dataset_obj, batch_size=1)

    right_dataset_obj = OpenNeuroCannabisUsersDataset(
        hemisphere="right", timepoint="baseline"
    )
    right_dataloader = PygDataLoader(right_dataset_obj, batch_size=1)

    left_src_graph, left_tgt_graph = next(iter(left_dataloader))
    right_src_graph, right_tgt_graph = next(iter(right_dataloader))

    assert not torch.equal(left_src_graph.x, right_src_graph.x)
    assert not torch.equal(left_tgt_graph.x, right_tgt_graph.x)


def test_openneuro_timepoints() -> None:
    """Test if openneuro dataset takes graphs on timepoints correctly."""
    bl_dataset_obj = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline"
    )
    bl_dataloader = PygDataLoader(bl_dataset_obj, batch_size=1)

    fu_dataset_obj = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="followup"
    )
    fu_dataloader = PygDataLoader(fu_dataset_obj, batch_size=1)

    fu_src_graph, fu_tgt_graph = next(iter(fu_dataloader))
    bl_src_graph, bl_tgt_graph = next(iter(bl_dataloader))

    assert not torch.equal(bl_tgt_graph.x, fu_tgt_graph.x)
    assert not torch.equal(bl_src_graph.x, fu_src_graph.x)


def test_cross_validation() -> None:
    """Test if cross validation splits work correctly."""
    tr_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline", mode="train"
    )
    val_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline", mode="validation"
    )
    assert tr_dataset.n_subj == len(tr_dataset.tr_indices)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_labels)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_nodes)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_edges)

    tr_set = set(tr_dataset.tr_indices)
    val_set = set(tr_dataset.val_indices)
    intersect = tr_set.intersection(val_set)
    assert len(intersect) == 0

    tr_set = set(val_dataset.tr_indices)
    val_set = set(val_dataset.val_indices)
    intersect = tr_set.intersection(val_set)
    assert len(intersect) == 0


def test_view_selection() -> None:
    """Test if view selection works correctly."""
    n_views = 4
    n_nodes = 34
    tr_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline", mode="train", in_view_idx=0
    )
    tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    src_graph, tgt_graph = next(iter(tr_dataloader))
    assert src_graph.x.shape == (1, n_nodes, 1)
    assert src_graph.edge_attr.shape == (1, n_nodes * n_nodes, 1)
    assert tgt_graph.x.shape == (1, n_nodes, n_views)
    assert tgt_graph.edge_attr.shape == (1, n_nodes * n_nodes, n_views)

    tr_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left",
        timepoint="baseline",
        mode="train",
        in_view_idx=0,
        out_view_idx=3,
    )
    tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    src_graph, tgt_graph = next(iter(tr_dataloader))
    assert src_graph.x.shape == (1, n_nodes, 1)
    assert src_graph.edge_attr.shape == (1, n_nodes * n_nodes, 1)
    assert tgt_graph.x.shape == (1, n_nodes, 1)
    assert tgt_graph.edge_attr.shape == (1, n_nodes * n_nodes, 1)
    assert not torch.equal(src_graph.x, tgt_graph.x)
    assert not torch.equal(src_graph.edge_attr, tgt_graph.edge_attr)
