"""Freesurfer utilities necessary for aparcstats2table module."""

from __future__ import annotations

# import logging
from typing import Any

from avicortex.freesurfer.types import Ddict, StableDict


def unique_union(seq: list[Any]) -> list[Any]:
    """Given a sequence, return a sequence with unique items with order intact."""
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def intersect_order(s1: list[Any], s2: list[Any]) -> list[Any]:
    """Given 2 sequences return the intersection with order intact as much as possible."""
    seen = {}
    result = []
    for item in s1:
        if item in seen:
            continue
        seen[item] = 1
    for item in s2:
        if item not in seen:
            continue
        result.append(item)
    return result


def make_table2d(
    disorganized_table: list[tuple[str, dict[str, Any]]], parcslist: list[str]
) -> tuple[list[str], list[str], Ddict]:
    """
    Reconstruct the stats as a matrix to prepare them to put into a table.

    Parameters
    ----------
        disorganized_table - the table is of the form (specifier, parc_measure_map)
        parcslist - list of parcellation names
        where parc_measure_map is a stable hashtable of keys parcellation names and values the measures.
        The table is disorganized because the length of the parc_measure_map may not be the same for all
        specifiers.
        parcellations present in parcslist are the only parcellations which go in the table.
        if any specifier doesn't have a parcellation, the measure is 0.0

    Returns
    -------
        rows - list of specifiers ( subjects)
        columns - list of parcellation names
        table - a stable 2d table of size len(rows) x len(columns)
    """
    dt = disorganized_table

    # create an ordered 2d table
    table = Ddict(StableDict)
    for _spec, _parc_measure_map in dt:
        for parc in parcslist:
            try:
                table[_spec][parc] = _parc_measure_map[parc]
            except KeyError:
                table[_spec][parc] = 0.0

    return [spec for (spec, i) in dt], parcslist, table


"""
Args:
    parsed options
    disorganized_table - the table is of the form (specifier, parc_measure_map)
    where parc_measure_map is a stable hashtable of keys parcellation names and values the measures.
    The table is disorganized because the length of the parc_measure_map may not be the same for all
    specifiers.
Returns:
    rows - list of specifiers ( subjects)
    columns - list of parcellation names
    table - a stable 2d table of size len(rows) x len(columns)
"""


def format_column_names(columns: list[str], pretext: str, posttext: str) -> list[str]:
    """Format column names."""
    cols = []
    for c in columns:
        if c in {"eTIV", "BrainSegVolNotVent"}:
            # For eTIV in aparc stats file
            cols.append(c)
        else:
            cols.append(pretext + c + posttext)
    return cols


def sanitize_table(
    disorganized_table: list[tuple[str, dict[str, Any]]], commonparcflag: bool = True
) -> tuple[list[str], list[str], Ddict]:
    """Organize and clean data from the stats table."""
    _union = []
    _, _parc_measure_map = disorganized_table[0]
    intersection = list(_parc_measure_map.keys())
    for _, parc_measure_map in disorganized_table:
        parcs = list(parc_measure_map.keys())
        _union.append(parcs)
        intersection = intersect_order(intersection, parcs)
        # l.debug("-" * 20)
        # l.debug("Specifier: " + spec)
        # l.debug("Intersection upto now:")
        # l.debug(intersection)
    # _union is a list of lists. Make it a flat list ( single list )
    temp_union = [item for sublist in _union for item in sublist]
    union = unique_union(temp_union)
    # l.debug("-" * 20)
    # l.debug("Union:")
    # l.debug(union)

    if commonparcflag:
        # write only the common parcs ( intersection )
        row, column, table = make_table2d(disorganized_table, intersection)
    else:
        # write all the parcs ever encountered
        # if there's no parcs for a certain .stats file, write the measure as 0.0
        row, column, table = make_table2d(disorganized_table, union)

    return row, column, table
