"""Test graph dataset classes."""
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
    src_graph, _ = next(iter(dataloader))
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


def test_hemispheres() -> None:
    """Test if the dataset can read different hemispheres correctly."""
    pass


def test_openneuro_timepoints() -> None:
    """Test if openneuro dataset takes graphs on timepoints correctly."""
    # n_views = 4
    # n_nodes = 34
    # baseline_dataset = OpenNeuroCannabisUsersDataset(
    #     hemisphere="left", timepoint="baseline"
    # )
    # followup_dataset = OpenNeuroCannabisUsersDataset(
    #     hemisphere="left", timepoint="followup"
    # )


def test_cross_validation() -> None:
    """Test if cross validation splits work correctly."""
    tr_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline", mode="train"
    )
    val_dataset = OpenNeuroCannabisUsersDataset(
        hemisphere="left", timepoint="baseline", mode="validation"
    )
    tr_dataloader = PygDataLoader(tr_dataset, batch_size=1)
    val_dataloader = PygDataLoader(val_dataset, batch_size=1)
    assert tr_dataset.n_subj == len(tr_dataset.tr_indices)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_labels)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_nodes)
    assert tr_dataset.n_subj == len(tr_dataset.subjects_edges)
    tr_sample = next(iter(tr_dataloader))
    val_sample = next(iter(val_dataloader))
    assert tr_sample is not None
    assert val_sample is not None
