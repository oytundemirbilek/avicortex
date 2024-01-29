"""Freesurfer utilities necessary for aparcstats2table module."""
from __future__ import annotations

import logging
import os
import sys
from typing import Any
from warnings import warn as _warn


# optparse can't handle variable number of arguments for an option.
# this callback allows that.
def callback_var(option, opt_str, value, parser):
    value = []
    rargs = parser.rargs
    while rargs:
        arg = rargs[0]
        if (arg[:2] == "--" and len(arg) > 2) or (
            arg[:1] == "-" and len(arg) > 1 and arg[1] != "-"
        ):
            break
        else:
            value.append(arg)
            del rargs[0]
    setattr(parser.values, option.dest, value)


def check_subjdirs() -> str:
    """
    Quit if SUBJECTS_DIR is not defined as an environment variable.

    This is not a function which returns a boolean. Execution is stopped if not found.
    If found, returns the SUBJECTS_DIR
    """
    if "SUBJECTS_DIR" not in os.environ:
        print("ERROR: SUBJECTS_DIR environment variable not defined!")
        sys.exit(1)
    return os.environ["SUBJECTS_DIR"]


class Ddict(dict):
    """
    Datastructure is used to store 2d Table.

    Mainly for a*stats2table
    Usage:
    >>> tb = Ddict( dict )
    >>> tb['bert']['putamen'] = .05
    >>> tb['bert']['caudate'] = 1.6
    >>> tb['fsaverage']['putamen'] = 2.2
    >>> car_details
    {'fsaverage': {'putamen': 2.2}, 'bert': {'putamen': 0.05, 'caudate':
        1.6}}
    """

    def __init__(self, default: Any = None) -> None:
        self.default = default

    def __getitem__(self, key: str) -> Any:
        if key not in self:
            self[key] = self.default()
        return dict.__getitem__(self, key)


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

    def __init__(self, r, c, table) -> None:
        self.rows = r
        self.columns = c
        self.table = table
        self.pretext = ""
        self.posttext = ""

    def assign_attributes(
        self, filename: str = "stats.table", row1col1: str = "", delimiter: str = "\t"
    ) -> None:
        self.filename = filename
        self.row1col1 = row1col1
        self.delimiter = delimiter

    def decorate_col_titles(self, pretext: str, posttext: str) -> None:
        self.pretext = pretext
        self.posttext = posttext

    def write(self) -> None:
        fp = open(self.filename, "w")
        fp.write(self.row1col1)
        for c in self.columns:
            if (c == "eTIV" or c == "BrainSegVolNotVent") and (
                self.pretext == "lh_" or self.pretext == "rh_"
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
        fp.close()

    def write_transpose(self) -> None:
        fp = open(self.filename, "w")
        fp.write(self.row1col1)
        for r in self.rows:
            fp.write(self.delimiter + r)
        fp.write("\n")

        for c in self.columns:
            if (c == "eTIV" or c == "BrainSegVolNotVent") and (
                self.pretext == "lh_" or self.pretext == "rh_"
            ):
                # For eTIV in aparc stats file
                fp.write(c)
            else:
                fp.write(self.pretext + c + self.posttext)
            for r in self.rows:
                fp.write(self.delimiter + "%g" % self.table[r][c])
            fp.write("\n")
        fp.close()


"""
A dictionary class remembering insertion order.

Order (i.e. the sequence) of insertions is remembered (internally
stored in a hidden list attribute) and replayed when iterating. A
StableDict does NOT sort or organize the keys in any other way.
"""


# Helper metaclass-function.  Not exported by default but accessible
# as StableDict.__metaclass__.
def copy_baseclass_docs(classname, bases, dict, metaclass=type):
    """Copy docstrings from baseclass.

    When overriding methods in a derived class the docstrings can
    frequently be copied from the base class unmodified.  According to
    the DRY principle (Don't Repeat Yourself) this should be
    automated. Putting a reference to this function into the
    __metaclass__ slot of a derived class will automatically copy
    docstrings from the base classes for all doc-less members of the
    derived class.
    """
    for name, member in dict.iteritems():
        if getattr(member, "__doc__", None):
            continue
        for base in bases:  # look only in direct ancestors
            basemember = getattr(base, name, None)
            if not basemember:
                continue
            basememberdoc = getattr(basemember, "__doc__", None)
            if basememberdoc:
                member.__doc__ = basememberdoc
    return metaclass(classname, bases, dict)


# String constants for Exceptions / Warnings:
_ERRsizeChanged = "StableDict changed size during iteration!"
_WRNnoOrderArg = "StableDict created/updated from unordered mapping object"
_WRNnoOrderKW = "StableDict created/updated with (unordered!) keyword arguments"


# Note: This class won't work with Python 3000 because the dict
#       protocol will change according to PEP3106. (However porting it
#       to Python 3.X will not be much of an effort.)
class StableDict(dict):
    """Dictionary remembering insertion order.

    Order of item assignment is preserved and repeated when iterating
    over an instance.

    CAVEAT: When handing an unordered dict to either the constructor
    or the update() method the resulting order is obviously
    undefined. The same applies when initializing or updating with
    keyword arguments; i.e. keyword argument order is not preserved. A
    runtime warning will be issued in these cases via the
    warnings.warn function.
    """

    __metaclass__ = copy_baseclass_docs  # copy docstrings from base class

    # Python 2.2 does not mangle __* inside __slots__
    __slots__ = ("_StableDict__ksl",)  # key sequence list aka __ksl

    # @staticmethod
    def is_ordered(dictInstance):
        """Return true if argument is known to be ordered."""
        if isinstance(dictInstance, StableDict):
            return True
        try:  # len() may raise an exception.
            if len(dictInstance) <= 1:
                return True  # A length <= 1 implies ordering.
        except:
            pass
        return False  # Assume dictInstance.keys() is _not_ ordered.

    is_ordered = staticmethod(is_ordered)

    def __init__(self, *arg, **kw):
        if arg:
            if len(arg) > 1:
                raise TypeError("at most one argument permitted")
            arg = arg[0]
            if hasattr(arg, "keys"):
                if not self.is_ordered(arg):
                    _warn(_WRNnoOrderArg, RuntimeWarning, stacklevel=2)
                super().__init__(arg, **kw)
                self.__ksl = list(arg.keys())
            else:  # Must be a sequence of 2-tuples.
                super().__init__(**kw)
                self.__ksl = []
                for pair in arg:
                    if len(pair) != 2:
                        raise ValueError("not a 2-tuple", pair)
                    self.__setitem__(pair[0], pair[1])
                if kw:
                    ksl = self.__ksl
                    for k in super().iterkeys():
                        if k not in ksl:
                            ksl.append(k)
                    self.__ksl = ksl
        else:  # No positional argument given.
            super().__init__(**kw)
            self.__ksl = list(super().keys())
        if len(kw) > 1:
            # There have been additionial keyword arguments.
            # Since Python passes them in an (unordered) dict
            # we cannot possibly preserve their order (without
            # inspecting the source or byte code of the call).
            _warn(_WRNnoOrderKW, RuntimeWarning, stacklevel=2)

    def update(self, *arg, **kw):
        if arg:
            if len(arg) > 1:
                raise TypeError("at most one non-keyword argument permitted")
            arg = arg[0]
            if hasattr(arg, "keys"):
                if not self.is_ordered(arg):
                    _warn(_WRNnoOrderArg, RuntimeWarning, stacklevel=2)
                super().update(arg)
                ksl = self.__ksl
                for k in arg.keys():
                    if k not in ksl:
                        ksl.append(k)
                self.__ksl = ksl
            else:  # Must be a sequence of 2-tuples.
                for pair in arg:
                    if len(pair) != 2:
                        raise ValueError("not a 2-tuple", pair)
                    self.__setitem__(pair[0], pair[1])
        if kw:
            # There have been additionial keyword arguments.
            # Since Python passes them in an (unordered) dict
            # we cannot possibly preserve their order (without
            # inspecting the source or byte code of the call).
            if len(kw) > 1:
                _warn(_WRNnoOrderKW, RuntimeWarning, stacklevel=2)
            super().update(kw)
            ksl = self.__ksl
            for k in kw.iterkeys():
                if k not in ksl:
                    ksl.append(k)
            self.__ksl = ksl

    def __str__(self):
        def _repr(x):
            if x is self:
                return "StableDict({...})"  # Avoid unbounded recursion.
            return repr(x)

        return (
            "StableDict({"
            + ", ".join(["{!r}: {}".format(k, _repr(v)) for k, v in self.iteritems()])
            + "})"
        )

    # Try to achieve: self == eval(repr(self))
    def __repr__(self):
        def _repr(x):
            if x is self:
                return "StableDict({...})"  # Avoid unbounded recursion.
            return repr(x)

        return (
            "StableDict(["
            + ", ".join(["({!r}, {})".format(k, _repr(v)) for k, v in self.iteritems()])
            + "])"
        )

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key not in self.__ksl:
            self.__ksl.append(key)

    def __delitem__(self, key):
        if key in self.__ksl:
            self.__ksl.remove(key)
        super().__delitem__(key)

    def __iter__(self):
        length = len(self)
        yield from self.__ksl[:]
        if length != len(self):
            raise RuntimeError(_ERRsizeChanged)

    def keys(self):
        return self.__ksl[:]

    def iterkeys(self):
        return self.__iter__()

    def values(self):
        return [self[k] for k in self.__ksl]

    def itervalues(self):
        length = len(self)
        for key in self.__ksl[:]:
            yield self[key]
        if length != len(self):
            raise RuntimeError(_ERRsizeChanged)

    def items(self):
        return [(k, self[k]) for k in self.__ksl]

    def iteritems(self):
        length = len(self)
        for key in self.__ksl[:]:
            yield (key, self[key])
        if length != len(self):
            raise RuntimeError(_ERRsizeChanged)

    def clear(self) -> None:
        """"""
        super().clear()
        self.__ksl = []

    def copy(self) -> StableDict:
        """"""
        return StableDict(self)

    def pop(self, k: str, *default: Any) -> Any:
        """"""
        if k in self.__ksl:
            self.__ksl.remove(k)
        return super().pop(k, *default)

    def popitem(self) -> Any:
        """"""
        item = super().popitem()
        try:
            self.__ksl.remove(item[0])
        except:
            raise ValueError("cannot remove", item, self.__ksl, self)
        return item


# Our metaclass function became a method.  Make it a function again.
StableDict.__metaclass__ = staticmethod(copy_baseclass_docs)


# Given a sequence, return a sequence with unique items with order intact
def unique_union(seq):
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


# Given 2 sequences return the intersection with order intact as much as
# possible
def intersect_order(s1, s2):
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


ch = logging.StreamHandler()
aparclogger = logging.getLogger("aparcstats2table")
aparclogger.setLevel(logging.INFO)
aparclogger.addHandler(ch)


class BadFileError(Exception):
    def __init__(self, filename):
        self.filename = filename

    def __str__(self):
        return self.filename


"""
This is the base class which parses the .stats files.
"""


class StatsParser:
    """Base class for other stats parsers."""

    measure_column_map = {}  # to be defined in subclass
    fp = None  # filepointer
    include_structlist = None  # parse only these structs
    exclude_structlist = None  # exclude these structs

    structlist = None  # list of structures
    measurelist = None  # list of measures corresponding to the structures
    # len ( structlist ) must be equal to len (measurelist )

    # constructor just needs a .stats filename
    # load it and report error
    def __init__(self, filename: str) -> None:
        self.filename = filename
        # raise exception if file doesn't exist or
        # is too small to be a valid stats file
        if not os.path.exists(filename):
            raise BadFileError(filename)
        if os.path.getsize(filename) < 10:
            raise BadFileError(filename)
        self.fp = open(filename)

        self.include_structlist = []
        self.exclude_structlist = []
        self.structlist = []
        self.measurelist = []
        self.include_vol_extras = 1

    # parse only the following structures
    def parse_only(self, structlist) -> None:
        # this is simply an Ordered Set
        # we need this because if inputs repeat, this will take care
        self.include_structlist = StableDict()
        for struct in structlist:
            self.include_structlist[struct] = 1
        self.structlist = []
        self.measurelist = []

    # exclude the following structures
    def exclude_structs(self, structlist) -> None:
        # this is simply an Ordered Set
        # we need this because if inputs repeat, this will take care
        self.exclude_structlist = StableDict()
        for struct in structlist:
            self.exclude_structlist[struct] = 1
        self.structlist = []
        self.measurelist = []

    # actual parsing will be done by subclass
    def parse(self):
        pass


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
    def parse(self, measure):
        self.parc_measure_map = StableDict()
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
            for oparc in self.include_structlist.keys():
                parclist = self.parc_measure_map.keys()
                if oparc in parclist:
                    tmp_parc_measure_map[oparc] = self.parc_measure_map[oparc]
                else:
                    tmp_parc_measure_map[oparc] = 0.0
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
