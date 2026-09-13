"""Source registry.
Adding a board means writing one module with fetch() and to_posting(), then
adding it here. Nothing else in the pipeline changes.
"""

from joblens.sources import adzuna, hackernews, remoteok

REGISTRY = {
    remoteok.name: remoteok,
    hackernews.name: hackernews,
    adzuna.name: adzuna,
}
DEFAULT_SOURCES = [remoteok.name, hackernews.name, adzuna.name]


def get(source_name: str):
    try:
        return REGISTRY[source_name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"unknown source {source_name!r}. known sources: {known}"
        ) from None
