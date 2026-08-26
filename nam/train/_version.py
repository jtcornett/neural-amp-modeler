# File: _version.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Version utility
"""

from .._version import __version__ as _package_version


class Version:
    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    @classmethod
    def from_string(cls, s: str):
        import re

        parts = s.split(".")
        nums = []
        for p in parts[:3]:
            match = re.search(r"\d+", p)
            nums.append(int(match.group()) if match else 0)
        while len(nums) < 3:
            nums.append(0)
        return cls(major=nums[0], minor=nums[1], patch=nums[2])

    def __eq__(self, other) -> bool:
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
        )

    def __lt__(self, other) -> bool:
        if self == other:
            return False
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        raise RuntimeError(
            f"Unhandled comparison between versions {str(self)} and {str(other)}"
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


PROTEUS_VERSION = Version(4, 0, 0)


def get_current_version() -> Version:
    return Version.from_string(_package_version)
