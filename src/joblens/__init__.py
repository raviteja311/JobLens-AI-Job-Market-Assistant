"""JobLens: AI job market assistant."""

from importlib.metadata import PackageNotFoundError, version

# One version, the one in pyproject.toml. The API used to hard-code its own
# and the two had drifted to 0.4.0 and 0.1.0. A checkout that was never
# pip-installed has no package metadata, and says so rather than guessing.
try:
    __version__ = version("joblens")
except PackageNotFoundError:
    __version__ = "0+unknown"
