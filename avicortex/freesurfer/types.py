"""Module to define special data types for freesurfer."""

from typing import Any
from warnings import warn as _warn

from _collections_abc import Generator, dict_items, dict_keys, dict_values

from avicortex.freesurfer.exceptions import (
    _ERRsizeChanged,
    _WRNnoOrderArg,
    _WRNnoOrderKW,
)


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
        """Access an element in the dictionary with the key."""
        if key not in self:
            self[key] = self.default()
        return dict.__getitem__(self, key)


class StableDict(dict):
    """Dictionary remembering insertion order.

    Order (i.e. the sequence) of insertions is remembered (internally
    stored in a hidden list attribute) and repeated when iterating
    over an instance. A StableDict does NOT sort or organize the keys
    in any other way.

    CAVEAT: When handing an unordered dict to either the constructor
    or the update() method the resulting order is obviously
    undefined. The same applies when initializing or updating with
    keyword arguments; i.e. keyword argument order is not preserved. A
    runtime warning will be issued in these cases via the
    warnings.warn function.
    """

    # Python 2.2 does not mangle __* inside __slots__
    __slots__ = ("_StableDict__ksl",)  # key sequence list aka __ksl

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            if len(args) > 1:
                raise TypeError("at most one argument permitted")
            first_arg = args[0]
            if hasattr(first_arg, "keys"):
                if not self.is_ordered(first_arg):
                    _warn(_WRNnoOrderArg, RuntimeWarning, stacklevel=2)
                super().__init__(first_arg, **kwargs)
                self.__ksl = list(first_arg.keys())
            else:  # Must be a sequence of 2-tuples.
                super().__init__(**kwargs)
                self.__ksl = []
                for pair in first_arg:
                    if len(pair) != 2:
                        raise ValueError("not a 2-tuple", pair)
                    self.__setitem__(pair[0], pair[1])
                if kwargs:
                    ksl = self.__ksl
                    for k in super().iterkeys():
                        if k not in ksl:
                            ksl.append(k)
                    self.__ksl = ksl
        else:  # No positional argument given.
            super().__init__(**kwargs)
            self.__ksl = list(super().keys())
        if len(kwargs) > 1:
            # There have been additionial keyword arguments.
            # Since Python passes them in an (unordered) dict
            # we cannot possibly preserve their order (without
            # inspecting the source or byte code of the call).
            _warn(_WRNnoOrderKW, RuntimeWarning, stacklevel=2)

    @staticmethod
    def is_ordered(dict_instance: dict) -> bool:
        """Return true if argument is known to be ordered."""
        if isinstance(dict_instance, StableDict):
            return True
        try:  # len() may raise an exception.
            if len(dict_instance) <= 1:
                return True  # A length <= 1 implies ordering.
        except Exception as e:
            raise e
        return False  # Assume dictInstance.keys() is _not_ ordered.

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update dictionary with provided args kwargs."""
        max_len_pair = 2
        if args:
            if len(args) > 1:
                raise TypeError("at most one non-keyword argument permitted")
            args = args[0]
            if hasattr(args, "keys"):
                if not self.is_ordered(args):
                    _warn(_WRNnoOrderArg, RuntimeWarning, stacklevel=2)
                super().update(args)
                ksl = self.__ksl
                for k in args:
                    if k not in ksl:
                        ksl.append(k)
                self.__ksl = ksl
            else:  # Must be a sequence of 2-tuples.
                for pair in args:
                    if len(pair) != max_len_pair:
                        raise ValueError("not a 2-tuple", pair)
                    self[pair[0]] = pair[1]
        if kwargs:
            # There have been additionial keyword arguments.
            # Since Python passes them in an (unordered) dict
            # we cannot possibly preserve their order (without
            # inspecting the source or byte code of the call).
            if len(kwargs) > 1:
                _warn(_WRNnoOrderKW, RuntimeWarning, stacklevel=2)
            super().update(kwargs)
            ksl = self.__ksl
            for k in kwargs.iterkeys():
                if k not in ksl:
                    ksl.append(k)
            self.__ksl = ksl

    def __str__(self) -> str:
        """Return string representation of the class."""

        def _repr(x: "StableDict") -> str:
            if x is self:
                return "StableDict({...})"  # Avoid unbounded recursion.
            return repr(x)

        return (
            "StableDict({"
            + ", ".join([f"{k!r}: {_repr(v)}" for k, v in self.iteritems()])
            + "})"
        )

    def __repr__(self) -> str:
        """Return string representation of the class."""

        def _repr(x: "StableDict") -> str:
            if x is self:
                return "StableDict({...})"  # Avoid unbounded recursion.
            return repr(x)

        return (
            "StableDict(["
            + ", ".join([f"({k!r}, {_repr(v)})" for k, v in self.iteritems()])
            + "])"
        )

    def __setitem__(self, key: str, value: Any) -> None:
        """Assign the input value to the provided key."""
        super().__setitem__(key, value)
        if key not in self.__ksl:
            self.__ksl.append(key)

    def __delitem__(self, key: str) -> None:
        """Delete item with the key."""
        if key in self.__ksl:
            self.__ksl.remove(key)
        super().__delitem__(key)

    def __iter__(self) -> Generator[list[Any], None, None]:
        """Grab next item in the class."""
        length = len(self)
        yield from self.__ksl[:]
        if length != len(self):
            raise RuntimeError(_ERRsizeChanged)

    def keys(self) -> dict_keys:
        """Return list of keys."""
        return self.__ksl[:]

    def iterkeys(self) -> Generator[list[Any], None, None]:
        """Grab next item in the class."""
        return iter(self)

    def values(self) -> dict_values:
        """Return values."""
        return [self[k] for k in self.__ksl]

    def itervalues(self) -> Generator[Any, None, None]:
        """Get next value."""
        length = len(self)
        for key in self.__ksl[:]:
            yield self[key]
        if length != len(self):
            raise RuntimeError(_ERRsizeChanged)

    def items(self) -> dict_items:
        """Get all key-value pairs."""
        return [(k, self[k]) for k in self.__ksl]

    def iteritems(self) -> Generator[tuple[str, Any], None, None]:
        """Yield all key-value pairs."""
        length = len(self)
        for key in self.__ksl[:]:
            yield (key, self[key])
        if length != len(self):
            raise RuntimeError(_ERRsizeChanged)

    def clear(self) -> None:
        """Empty dictionary."""
        super().clear()
        self.__ksl = []

    def copy(self) -> "StableDict":
        """Create a new object and return it."""
        return StableDict(self)

    def pop(self, k: str, *default: Any) -> Any:
        """Remove and return the element with the key."""
        if k in self.__ksl:
            self.__ksl.remove(k)
        return super().pop(k, *default)

    def popitem(self) -> Any:
        """Remove and return first element."""
        item = super().popitem()
        try:
            self.__ksl.remove(item[0])
        except Exception as e:
            raise ValueError("cannot remove", item, self.__ksl, self) from e
        return item