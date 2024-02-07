"""Module to define exception classes."""

# String constants for Exceptions / Warnings:
_ERRsizeChanged = "StableDict changed size during iteration!"
_WRNnoOrderArg = "StableDict created/updated from unordered mapping object"
_WRNnoOrderKW = "StableDict created/updated with (unordered!) keyword arguments"


class BadFileError(Exception):
    """Exception raised if the target file is not in expected format."""

    def __init__(self, filename: str) -> None:
        self.filename = filename

    def __str__(self) -> str:
        """Return string representation of the erroneous file."""
        return self.filename
