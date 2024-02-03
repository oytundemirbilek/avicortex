"""Module for write operations."""

from __future__ import annotations

from avicortex.freesurfer.types import Ddict


class TableWriter:
    """
    Class writes a 2d Table of type Ddict(dict) to a file.

    Some essential things needs to be set for this class
    rows - a sequence of text which go in the first column
    columns - a sequence of text which go in the first row
    table - a Ddict(dict) object which has *exactly* len(columns) x len(rows) elements
    row1col1 - the text which goes in 1st row and 1st column
    delimiter - what separates each element ( default is a tab )
    filename - the filename to write to.
    """

    def __init__(self, r: list[str], c: list[str], table: Ddict) -> None:
        self.rows = r
        self.columns = c
        self.table = table
        self.pretext = ""
        self.posttext = ""

    def assign_attributes(
        self, filename: str = "stats.table", row1col1: str = "", delimiter: str = ","
    ) -> None:
        """Define formatting of the table file."""
        self.filename = filename
        self.row1col1 = row1col1
        self.delimiter = delimiter

    def decorate_col_titles(self, pretext: str, posttext: str) -> None:
        """Define formatting of the column names."""
        self.pretext = pretext
        self.posttext = posttext

    def write(self) -> None:
        """Write table."""
        with open(self.filename, "w", encoding="utf-8") as fp:
            fp.write(self.row1col1)
            for c in self.columns:
                if (c in {"eTIV", "BrainSegVolNotVent"}) and (
                    self.pretext in {"lh", "rh"}
                ):
                    # For eTIV in aparc stats file
                    fp.write(self.delimiter + c)
                else:
                    fp.write(self.delimiter + self.pretext + c + self.posttext)
            fp.write("\n")

            for r in self.rows:
                fp.write(r)
                for c in self.columns:
                    fp.write(self.delimiter + "%s" % self.table[r][c])
                fp.write("\n")

    def write_transpose(self) -> None:
        """Write transpose of the table."""
        with open(self.filename, "w", encoding="utf-8") as fp:
            fp.write(self.row1col1)
            for r in self.rows:
                fp.write(self.delimiter + r)
            fp.write("\n")

            for c in self.columns:
                if (c in {"eTIV", "BrainSegVolNotVent"}) and (
                    self.pretext in {"lh", "rh"}
                ):
                    # For eTIV in aparc stats file
                    fp.write(c)
                else:
                    fp.write(self.pretext + c + self.posttext)
                for r in self.rows:
                    fp.write(self.delimiter + "%g" % self.table[r][c])
                fp.write("\n")
