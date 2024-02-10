"""Test graph dataset classes."""

import os

import torch
from torch_geometric.loader import DataLoader as PygDataLoader

from avicortex.datasets import ADNIAlzheimersDataset, OpenNeuroCannabisUsersDataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "mock_datasets")


def test_simple_iteration() -> None:
    """Test if a dataset can be iterated."""
    n_views = 5
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
    data_length = 42
    bl_dataset_obj = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline"
    )
    bl_dataloader = PygDataLoader(bl_dataset_obj, batch_size=1)
    assert len(bl_dataloader) == data_length

    fu_dataset_obj = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="followup"
    )
    fu_dataloader = PygDataLoader(fu_dataset_obj, batch_size=1)
    assert len(fu_dataloader) == data_length

    all_dataset_obj = OpenNeuroCannabisUsersDataset(hemisphere="left")
    all_dataloader = PygDataLoader(all_dataset_obj, batch_size=1)
    assert len(all_dataloader) == data_length * 2

    fu_src_graph, fu_tgt_graph = next(iter(fu_dataloader))
    bl_src_graph, bl_tgt_graph = next(iter(bl_dataloader))
    all_src_graph, all_tgt_graph = next(iter(all_dataloader))

    assert all_src_graph.x is not None
    assert all_tgt_graph.x is not None

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
    test_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline", mode="test"
    )
    assert tr_dataset.n_subj == len(tr_dataset.tr_indices)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_labels)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_nodes)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_edges)
    assert tr_dataset.n_subj == 16

    assert val_dataset.n_subj == len(val_dataset.val_indices)
    assert val_dataset.n_subj == len(val_dataset.subjects_labels)
    assert val_dataset.n_subj == len(val_dataset.subjects_nodes)
    assert val_dataset.n_subj == len(val_dataset.subjects_edges)
    assert val_dataset.n_subj == 5

    assert test_dataset.n_subj == len(test_dataset.unseen_data_indices)
    assert test_dataset.n_subj == len(test_dataset.subjects_labels)
    assert test_dataset.n_subj == len(test_dataset.subjects_nodes)
    assert test_dataset.n_subj == len(test_dataset.subjects_edges)
    assert test_dataset.n_subj == 21

    tr_set = set(tr_dataset.tr_indices)
    val_set = set(tr_dataset.val_indices)
    assert len(tr_set.intersection(val_set)) == 0

    tr_set = set(val_dataset.tr_indices)
    val_set = set(val_dataset.val_indices)
    assert len(tr_set.intersection(val_set)) == 0

    val_ids = set(val_dataset.subjects_ids)
    tr_ids = set(tr_dataset.subjects_ids)
    assert len(tr_ids.intersection(val_ids)) == 0

    seen_set = set(test_dataset.seen_data_indices)
    test_set = set(test_dataset.unseen_data_indices)
    assert len(test_set.intersection(seen_set)) == 0

    test_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint=None, mode="test"
    )
    assert test_dataset.n_subj == len(test_dataset.unseen_data_indices)
    assert test_dataset.n_subj == len(test_dataset.subjects_labels)
    assert test_dataset.n_subj == len(test_dataset.subjects_nodes)
    assert test_dataset.n_subj == len(test_dataset.subjects_edges)
    assert test_dataset.n_subj == 42
    seen_set = set(test_dataset.seen_data_indices)
    test_set = set(test_dataset.unseen_data_indices)
    assert len(test_set.intersection(seen_set)) == 0

    test_ids = set(test_dataset.subjects_ids)
    assert len(test_ids.intersection(tr_ids)) == 0
    assert len(test_ids.intersection(val_ids)) == 0


def test_data_split_ratios() -> None:
    """Test if different data split combinations work properly."""
    tr_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left",
        timepoint="baseline",
        mode="train",
        data_split_ratio=(9, 1, 5),
    )
    val_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left",
        timepoint="baseline",
        mode="validation",
        data_split_ratio=(9, 1, 5),
    )
    test_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left",
        timepoint="baseline",
        mode="test",
        data_split_ratio=(9, 1, 5),
    )
    assert tr_dataset.n_subj == 25
    assert val_dataset.n_subj == 3
    assert test_dataset.n_subj == 14

    # All test set
    test_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left",
        timepoint="baseline",
        mode="test",
        data_split_ratio=(0, 0, 1),
    )
    assert test_dataset.n_subj == 42

    # No test set
    tr_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left",
        timepoint="baseline",
        mode="train",
        data_split_ratio=(4, 1, 0),
    )
    assert tr_dataset.n_subj == 33

    val_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left",
        timepoint="baseline",
        mode="validation",
        data_split_ratio=(4, 1, 0),
    )
    assert val_dataset.n_subj == 9

    test_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left",
        timepoint="baseline",
        mode="test",
        data_split_ratio=(4, 1, 0),
    )
    assert test_dataset.n_subj == 0


def test_view_selection() -> None:
    """Test if view selection works correctly."""
    n_views = 5
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


def test_adni_dataset() -> None:
    """Test if ADNI dataset works correctly."""
    n_views = 3
    n_samples = 3
    n_nodes = 34
    tr_dataset = ADNIAlzheimersDataset(
        hemisphere="left",
        freesurfer_out_path=os.path.join(DATA_PATH, "adni", "adni_mock.csv"),
    )
    assert len(tr_dataset) == n_samples
    tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    src_graph, tgt_graph = next(iter(tr_dataloader))

    assert src_graph.x.shape == (1, n_nodes, n_views)
    assert src_graph.edge_attr.shape == (1, n_nodes * n_nodes, n_views)
    assert tgt_graph.x.shape == (1, n_nodes, n_views)
    assert tgt_graph.edge_attr.shape == (1, n_nodes * n_nodes, n_views)
