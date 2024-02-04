"""Test freesurfer utility functions."""

import os

import pandas as pd
from pandas.testing import assert_frame_equal

from avicortex.freesurfer.parsers import AparcStatsParser
from avicortex.freesurfer.reader import StatsCollector
from avicortex.freesurfer.types import StableDict

FILE_PATH = os.path.dirname(__file__)
UPDATE_GOLD_STANDARD = False


def test_stable_dict() -> None:
    """Test whether the StableDict class works properly."""
    sdict = StableDict()
    sdict["mock_key1"] = 1.0
    assert "mock_key1" in sdict

    sdict["mock_key0"] = 2.0
    sdict["mock_key3"] = 2.0
    sdict["mock_key2"] = 2.0
    assert list(sdict.keys())[0] == "mock_key1"
    assert list(sdict.keys())[1] == "mock_key0"

    dict_repr = sdict.to_dict()
    assert isinstance(dict_repr, dict)
    assert not isinstance(dict_repr, StableDict)
    assert "mock_key1" in dict_repr
    assert dict_repr["mock_key1"] == 1.0

    other_dict = {"mock_key1": 10.0, "mock_key10": 1.0}
    sdict.update(other_dict)
    assert "mock_key10" in sdict
    assert sdict["mock_key1"] == 10.0


def test_aparc_stats_read() -> None:
    """Test whether the parser reads the stats correctly."""
    subject_id = "sub-103"
    target_path = os.path.join(
        FILE_PATH,
        "test_subjects",
        "valid_subjects",
        subject_id,
        "stats",
        "lh.aparc.stats",
    )
    aparc_parser = AparcStatsParser(target_path)
    parc_measure_map = aparc_parser.parse(measure="meancurv")
    assert parc_measure_map is not None
    assert len(parc_measure_map) == 36
    assert "bankssts" in parc_measure_map
    assert isinstance(parc_measure_map["bankssts"], float)


def test_make_table() -> None:
    """Test whether make table function works properly."""


def test_stats_collector() -> None:
    """Test whether stats collector works properly."""
    target_path = os.path.join(FILE_PATH, "test_subjects", "valid_subjects")
    collector = StatsCollector(target_path)

    expected_path = os.path.join(
        FILE_PATH, "expected", "openneuro_baseline_dktatlas.csv"
    )
    expected_df = (
        pd.read_csv(expected_path).sort_values(by="Subject ID").reset_index(drop=True)
    )
    stats_df = (
        collector.collect_all().sort_values(by="Subject ID").reset_index(drop=True)
    )

    assert not stats_df.isna().any().any(), "Some NaN values."
    assert len(stats_df) == len(expected_df)
    # 7 measures, 2 hemispheres, 34 regions, some eTIV columns, 1 identifier column (7x2x34+1+eTIV)
    assert len(stats_df.columns) == 483
    if UPDATE_GOLD_STANDARD:
        stats_df.to_csv(expected_path, index=False)

    assert_frame_equal(stats_df, expected_df)


def test_stats_collector_regions() -> None:
    """Test whether stats collector collected all regions."""
