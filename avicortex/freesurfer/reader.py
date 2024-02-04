"""Module to collect stats for all subjects in the target folder."""

from __future__ import annotations

import os

import pandas as pd

from avicortex.freesurfer.exceptions import BadFileError
from avicortex.freesurfer.parsers import AparcStatsParser
from avicortex.freesurfer.utils import format_column_names, sanitize_table


class StatsCollectorError(Exception):
    """Common exceptions class for the stats collector module."""


class StatsCollector:
    """Base class to provide common functionalities to collect all stats."""

    def __init__(
        self,
        subjects_path: str,
        hemisphere: str = "both",
        measurements: list[str] | None = None,
        atlas: str = "dktatlas",
        skip_on_error: bool = True,
    ) -> None:
        self.subjects_path = subjects_path
        self.hemisphere = hemisphere
        if measurements is None:
            self.measurements = [
                "volume",
                "thickness",
                "area",
                "meancurv",
                "gauscurv",
                "foldind",
                "curvind",
            ]
        else:
            self.measurements = measurements
        self.atlas = atlas
        self.skip_on_error = skip_on_error

    def collect_measurement(self, measure: str, hem: str) -> pd.DataFrame:
        """Collect cortical region data from the specified measurement and hemisphere."""
        subjects = os.listdir(self.subjects_path)
        pretable = []
        for subject_id in subjects:
            filepath = os.path.join(
                self.subjects_path, subject_id, "stats", f"{hem}.aparc.stats"
            )
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
        _, _, table = sanitize_table(pretable, commonparcflag=False)
        df = pd.DataFrame.from_dict(table.to_dict(), orient="index")
        df.columns = format_column_names(df.columns, hem + "_", "_" + measure)
        return df

    def collect_hemisphere(self, hem: str) -> pd.DataFrame:
        """Collect cortical region data from multiple measurements and specified hemisphere."""
        data_list = []
        for idx, measure in enumerate(self.measurements):
            stats = self.collect_measurement(measure, hem)
            if idx != 0:
                stats = stats.drop(["eTIV", "BrainSegVolNotVent"], axis=1)
            data_list.append(stats)

        df = (
            pd.concat(data_list, axis=1, join="outer")
            .reset_index()
            .rename(columns={"index": "Subject ID"})
        )
        return df

    def collect_all(self) -> pd.DataFrame:
        """Collect all cortical attributes into one table."""
        if self.hemisphere == "both":
            return pd.concat(
                [
                    self.collect_hemisphere("lh"),
                    self.collect_hemisphere("rh").drop(
                        ["Subject ID", "eTIV", "BrainSegVolNotVent"], axis=1
                    ),
                ],
                axis=1,
                join="outer",
            )
        elif self.hemisphere == "left":
            return self.collect_hemisphere("lh")
        elif self.hemisphere == "right":
            return self.collect_hemisphere("rh")
        else:
            raise StatsCollectorError(
                "Invalid hemisphere name, should be 'left', 'right', or 'both'."
            )
