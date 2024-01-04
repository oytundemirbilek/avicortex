"""Tests graph builders."""

import os

from avicortex.builders import OpenNeuroGraphBuilder

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")


def test_builder_init() -> None:
    """Test if graph builder class initialization."""
    # Example run:
    # candi_path = os.path.join(DATA_PATH, "datasets", "candishare_schiz.csv")
    openneuro_bl_path = os.path.join(DATA_PATH, "openneuro_baseline_dktatlas.csv")
    # openneuro_fu_path = os.path.join(DATA_PATH, "openneuro_cannabis_users_followup.csv")

    # gbuilder = CandiShareGraphBuilder(candi_path)
    gbuilder = OpenNeuroGraphBuilder(openneuro_bl_path)
    labels = gbuilder.get_labels()
    nodes, edges = gbuilder.construct(hem="left")
    assert labels is not None
    assert nodes is not None
    assert edges is not None


def test_builder_labels() -> None:
    """Test if graph builder gets labels correctly."""


def test_builder_region_check() -> None:
    """Test if graph builder checks regions correctly."""
