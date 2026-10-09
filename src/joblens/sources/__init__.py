"""Source registry.
Adding a board means writing one module with fetch() and to_posting(), then
adding it here. Nothing else in the pipeline changes.

Hacker News and RemoteOK stay registered but out of the defaults: their
postings are mostly US and global, and keeping the modules lets the archived
v1 corpus (docs/archive/v1-global-corpus) still be re-parsed.
"""

from joblens.sources import ashby, greenhouse, hackernews, lever, remoteok

REGISTRY = {
    remoteok.name: remoteok,
    hackernews.name: hackernews,
    greenhouse.name: greenhouse,
    lever.name: lever,
    ashby.name: ashby,
}
DEFAULT_SOURCES = [greenhouse.name, lever.name, ashby.name]


def get(source_name: str):
    try:
        return REGISTRY[source_name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"unknown source {source_name!r}. known sources: {known}"
        ) from None
