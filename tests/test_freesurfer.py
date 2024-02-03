"""Test freesurfer utility functions."""

import logging
import os

from avicortex.freesurfer.parsers import AparcStatsParser

FILE_PATH = os.path.dirname(__file__)


def test_stable_dict() -> None:
    """Test whether the StableDict class works properly."""


def test_aparc_stats_read() -> None:
    """Test whether the parser reads the stats correctly."""
    subject_id = "sub-103"
    target_path = os.path.join(
        FILE_PATH, "test_subjects", subject_id, "stats", "lh.aparc.stats"
    )
    aparc_parser = AparcStatsParser(target_path)
    parc_measure_map = aparc_parser.parse(measure="meancurv")
    logging.info(parc_measure_map)

    pretable = [(subject_id, parc_measure_map)]
    logging.info(pretable)


def test_make_table() -> None:
    """Test whether make table function works properly."""


def test_stats_collector() -> None:
    """Test whether stats collector works properly."""
