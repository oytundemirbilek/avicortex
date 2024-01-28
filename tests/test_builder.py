"""Tests graph builders."""
import os

import pandas as pd

from avicortex.builders import OpenNeuroGraphBuilder

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "avicortex", "data")


def test_builder_init_timepoints() -> None:
    """Test if graph builder class initialization."""
    openneuro_bl_path = os.path.join(DATA_PATH, "openneuro_baseline_dktatlas.csv")
    openneuro_fu_path = os.path.join(DATA_PATH, "openneuro_followup_dktatlas.csv")

    gbuilder_bl = OpenNeuroGraphBuilder(openneuro_bl_path)
    gbuilder_fu = OpenNeuroGraphBuilder(openneuro_fu_path)

    assert gbuilder_bl.column_pattern is not None
    assert gbuilder_fu.column_pattern is not None

    assert gbuilder_bl.region_name_pos_in_pattern is not None
    assert gbuilder_fu.region_name_pos_in_pattern is not None

    assert gbuilder_bl.views is not None
    assert gbuilder_fu.views is not None

    assert gbuilder_bl.dkt_regions is not None
    assert gbuilder_fu.dkt_regions is not None

    assert gbuilder_bl.label_encoding is not None
    assert gbuilder_fu.label_encoding is not None


def test_builder_labels() -> None:
    """Test if graph builder gets labels correctly."""
    openneuro_bl_path = os.path.join(DATA_PATH, "openneuro_baseline_dktatlas.csv")

    gbuilder = OpenNeuroGraphBuilder(openneuro_bl_path)
    labels = gbuilder.get_labels()
    assert labels is not None
    assert labels.shape == (42,)


def test_builder_construct_graph() -> None:
    """Test if graph builder class initialization."""
    openneuro_bl_path = os.path.join(DATA_PATH, "openneuro_baseline_dktatlas.csv")

    gbuilder = OpenNeuroGraphBuilder(openneuro_bl_path)
    nodes, edges = gbuilder.construct(hem="left")
    assert nodes is not None
    assert edges is not None
    assert nodes.shape == (34, 42, 5)
    assert edges.shape == (34, 34, 42, 5)


def test_builder_region_check() -> None:
    """Test if graph builder checks regions correctly."""
    openneuro_bl_path = os.path.join(DATA_PATH, "openneuro_baseline_dktatlas.csv")

    gbuilder = OpenNeuroGraphBuilder(openneuro_bl_path)
    df_single_view = pd.DataFrame(
        columns=["lh_bankssts_area", "lh_caudalanteriorcingulate_area"]
    )
    not_dkt = gbuilder.check_regions(df_single_view)
    # Check for erroneous columns that does not indicate a DKT-atlas region.
    assert not_dkt is not None
    assert len(not_dkt) == 0

    df_single_view = pd.DataFrame(
        columns=["lh_bankssts_area", "lh_caudalanteriorcingulate_area", "not_a_region"]
    )
    not_dkt = gbuilder.check_regions(df_single_view)
    # Check for erroneous columns that does not indicate a DKT-atlas region.
    assert not_dkt is not None
    assert len(not_dkt) == 1
    assert not_dkt[0] == "not_a_region"
