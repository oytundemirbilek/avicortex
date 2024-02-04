"""Base module which parses the .stats files."""

from __future__ import annotations

import os
from typing import Any

from avicortex.freesurfer.exceptions import BadFileError
from avicortex.freesurfer.types import StableDict


class StatsParserError(Exception):
    """Raised when issues in StatsParser objects."""


class StatsParser:
    """Base class for other stats parsers."""

    measure_column_map: dict[str, Any] = {}  # to be defined in subclass
    fp = None  # filepointer
    include_structlist = None  # parse only these structs
    exclude_structlist = None  # exclude these structs

    structlist: list[str] | None = None  # list of structures
    measurelist: list[str] | None = None  # list of measures per structure
    # len ( structlist ) must be equal to len (measurelist )

    # constructor just needs a .stats filename
    # load it and report error
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.min_file_size = 10
        # raise exception if file doesn't exist or
        # is too small to be a valid stats file
        if not os.path.exists(filename):
            raise BadFileError(filename)
        if os.path.getsize(filename) < self.min_file_size:
            raise BadFileError(filename)
        self.fp = open(filename, encoding="utf-8")  # noqa: SIM115

        self.include_structlist = StableDict()
        self.exclude_structlist = StableDict()
        self.structlist = []
        self.measurelist = []
        self.include_vol_extras = 1

    # parse only the following structures
    def parse_only(self, structlist: list[str]) -> None:
        """Parse only the provided structures."""
        self.include_structlist = StableDict()
        for struct in structlist:
            self.include_structlist[struct] = 1
        self.structlist = []
        self.measurelist = []

    # exclude the following structures
    def exclude_structs(self, structlist: list[str]) -> None:
        """Exclude the following structures from parsing."""
        self.exclude_structlist = StableDict()
        for struct in structlist:
            self.exclude_structlist[struct] = 1
        self.structlist = []
        self.measurelist = []


class AparcStatsParser(StatsParser):
    """
    ?h.aparc*.stats parser ( or parser for similarly formatted .stats files ).

    Derived from StatsParser
    """

    # this is a map of measure requested and its corresponding column# in ?h.aparc*.stats
    measure_column_map = {
        "area": 2,
        "volume": 3,
        "thickness": 4,
        "thickness.T1": 4,
        "thicknessstd": 5,
        "meancurv": 6,
        "gauscurv": 7,
        "foldind": 8,
        "curvind": 9,
    }
    parc_measure_map = StableDict()

    # we take in the measure we need..
    def parse(self, measure: str) -> dict[str, Any]:  # noqa: C901
        """Parse the measurement."""
        self.parc_measure_map = StableDict()
        if self.fp is None:
            raise StatsParserError("File pointer is not valid.")
        for line in self.fp:
            # a valid line is a line without a '#'
            if line.rfind("#") == -1:
                strlist = line.split()
                # for every parcellation
                parcid = strlist[0]
                val = float(strlist[self.measure_column_map[measure]])
                self.parc_measure_map[parcid] = val

        # if we have a spec which instructs the table to have only specified parcs,
        # we need to make sure the order has to be maintained
        if self.include_structlist:
            tmp_parc_measure_map = StableDict()
            for oparc in self.include_structlist:
                tmp_parc_measure_map[oparc] = self.parc_measure_map.get(oparc, 0.0)
            self.parc_measure_map = tmp_parc_measure_map

        # measures which are found at the beginning of files.
        self.fp.seek(0)
        for line in self.fp:
            beg_struct_tuple = (
                ("# Measure EstimatedTotalIntraCranialVol, eTIV", "eTIV"),
            )
            for start, structn in beg_struct_tuple:
                if line.startswith(start):
                    strlst = line.split(",")
                    self.parc_measure_map[structn] = float(strlst[3])
            beg_struct_tuple = (
                ("# Measure BrainSegNotVent, BrainSegVolNotVent", "BrainSegVolNotVent"),
            )
            for start, structn in beg_struct_tuple:
                if line.startswith(start):
                    strlst = line.split(",")
                    self.parc_measure_map[structn] = float(strlst[3])
            if measure == "area":
                beg_struct_tuple = (
                    ("# Measure Cortex, WhiteSurfArea,", "WhiteSurfArea"),
                )
                for start, structn in beg_struct_tuple:
                    if line.startswith(start):
                        strlst = line.split(",")
                        self.parc_measure_map[structn] = float(strlst[3])
            if measure == "thickness":
                beg_struct_tuple = (
                    ("# Measure Cortex, MeanThickness,", "MeanThickness"),
                )
                for start, structn in beg_struct_tuple:
                    if line.startswith(start):
                        strlst = line.split(",")
                        self.parc_measure_map[structn] = float(strlst[3])

        return self.parc_measure_map
