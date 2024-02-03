"""Module to collect stats for all subjects in the target folder."""

import os

import pandas as pd

from avicortex.freesurfer.exceptions import BadFileError
from avicortex.freesurfer.parsers import AparcStatsParser


class StatsCollectorError(Exception):
    """Common exceptions class for the stats collector module."""


class StatsCollector:
    """Base class to provide common functionalities to collect all stats."""

    def __init__(
        self,
        subjects_path: str,
        hemisphere: str,
        measurements: list[str] | None = None,
        atlas: str = "dktatlas",
        skip_on_error: bool = True,
    ) -> None:
        self.subjects_path = subjects_path
        self.hemisphere = hemisphere
        if measurements is None:
            self.measurements = [""]
        else:
            self.measurements = measurements
        self.atlas = atlas
        self.skip_on_error = skip_on_error

    def collect_measurement(self, measure: str) -> pd.DataFrame:
        """"""
        subjects = os.listdir(self.subjects_path)
        pretable = []
        for subject_id in subjects:
            filepath = os.path.join(self.subjects_path, subject_id)
            try:
                parsed = AparcStatsParser(filepath)
                # parcs filter from the command line
                # if options.parcs is not None:
                #     parsed.parse_only(options.parcs)

                parc_measure_map = parsed.parse(measure)
            except BadFileError as e:
                if self.skip_on_error:
                    continue
                else:
                    raise StatsCollectorError(
                        "ERROR: The stats file is not found or is not a valid statsfile."
                    ) from e

            pretable.append((subject_id, parc_measure_map))

    def collect_all(self) -> pd.DataFrame:
        """"""
        data_list = []
        for measure in self.measurements:
            stats = self.collect_measurement(measure)
            data_list.append(stats)

        return pd.concat(data_list)
