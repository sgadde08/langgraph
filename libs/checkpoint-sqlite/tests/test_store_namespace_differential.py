"""Differential test of namespace matching against `InMemoryStore`.

`SqliteStore` flattens a namespace tuple into a dot-joined string and scopes reads
with `GLOB` and regular expressions, so the match semantics are re-implemented in
SQL. `InMemoryStore` compares namespaces element-wise as tuples, which is the
behavior the SQL stores are meant to reproduce.

Random namespaces are built from labels chosen to trip pattern matching (SQL
wildcards, brackets, regex metacharacters, case variants, shared prefixes), loaded
into both stores, and then every scoped read must return the same set of
namespaces. Namespaces are commonly used as a tenant boundary, so a divergence here
is a cross-tenant read.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Generator
from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore

from langgraph.store.sqlite import SqliteStore

# Labels that share prefixes (`a`/`ab`/`a_b`), carry glob/like/regex metacharacters,
# or differ only by case, so a flawed match rule reaches a sibling namespace.
LABELS = [
    "a",
    "ab",
    "A",
    "a_b",
    "aXb",
    "a%",
    "a%b",
    "a[b",
    "a]",
    "a?",
    "a*b",
    "a\\b",
    "a'b",
    'a"b',
    "a b",
    "a\nb",
    "1",
    "12",
    "10",
    "_",
    "%",
    "[",
    "]",
    "?",
    "^a",
    "a$",
    "(a)",
    "a|b",
    "é",
    "é",
    "i",
    "I",
]

SEEDS = range(12)
NAMESPACES_PER_SEED = 40
QUERIES_PER_SEED = 80


def _random_namespace(rng: random.Random) -> tuple[str, ...]:
    return tuple(rng.choice(LABELS) for _ in range(rng.randint(1, 3)))


def _namespaces(items: Any) -> list[tuple[str, ...]]:
    return sorted(tuple(namespace) for namespace in items)


@pytest.fixture
def store() -> Generator[SqliteStore, None, None]:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        yield store


@pytest.mark.parametrize("seed", SEEDS)
def test_namespace_matching_agrees_with_in_memory_store(
    store: SqliteStore, seed: int
) -> None:
    rng = random.Random(seed)
    reference = InMemoryStore()
    stored = sorted({_random_namespace(rng) for _ in range(NAMESPACES_PER_SEED)})
    for position, namespace in enumerate(stored):
        reference.put(namespace, "k", {"position": position})
        store.put(namespace, "k", {"position": position}, index=False)

    def pick() -> tuple[str, ...]:
        if rng.random() < 0.7:
            return rng.choice(stored)
        return _random_namespace(rng)

    def prefix_of(namespace: tuple[str, ...]) -> tuple[str, ...]:
        return namespace[: rng.randint(1, len(namespace))]

    def suffix_of(namespace: tuple[str, ...]) -> tuple[str, ...]:
        return namespace[-rng.randint(1, len(namespace)) :]

    def with_wildcard(namespace: tuple[str, ...]) -> tuple[str, ...]:
        parts = list(namespace)
        parts[rng.randrange(len(parts))] = "*"
        return tuple(parts)

    queries: list[tuple[str, Callable[[Any], Any]]] = []
    for _ in range(QUERIES_PER_SEED):
        namespace = pick()
        match rng.choice(["search", "prefix", "suffix", "wildcard", "both", "depth"]):
            case "search":
                scope = prefix_of(namespace)
                queries.append(
                    (
                        f"search(prefix={scope!r})",
                        lambda s, scope=scope: _namespaces(
                            item.namespace for item in s.search(scope, limit=1000)
                        ),
                    )
                )
            case "prefix":
                scope = prefix_of(namespace)
                queries.append(
                    (
                        f"list_namespaces(prefix={scope!r})",
                        lambda s, scope=scope: _namespaces(
                            s.list_namespaces(prefix=scope, limit=1000)
                        ),
                    )
                )
            case "suffix":
                scope = suffix_of(namespace)
                queries.append(
                    (
                        f"list_namespaces(suffix={scope!r})",
                        lambda s, scope=scope: _namespaces(
                            s.list_namespaces(suffix=scope, limit=1000)
                        ),
                    )
                )
            case "wildcard":
                scope = with_wildcard(namespace)
                queries.append(
                    (
                        f"list_namespaces(prefix={scope!r})",
                        lambda s, scope=scope: _namespaces(
                            s.list_namespaces(prefix=scope, limit=1000)
                        ),
                    )
                )
            case "both":
                head, tail = namespace[:1], namespace[-1:]
                queries.append(
                    (
                        f"list_namespaces(prefix={head!r}, suffix={tail!r})",
                        lambda s, head=head, tail=tail: _namespaces(
                            s.list_namespaces(prefix=head, suffix=tail, limit=1000)
                        ),
                    )
                )
            case _:
                depth = rng.randint(1, 3)
                queries.append(
                    (
                        f"list_namespaces(max_depth={depth})",
                        lambda s, depth=depth: _namespaces(
                            s.list_namespaces(max_depth=depth, limit=1000)
                        ),
                    )
                )

    for description, query in queries:
        assert query(store) == query(reference), f"seed={seed}: {description}"

    # `get` compares exact namespaces, so a near-miss label must not resolve.
    for namespace in stored:
        found = store.get(namespace, "k")
        assert found is not None and found.namespace == namespace
        assert store.get(namespace, "missing") is None
    absent = [
        ns for ns in (_random_namespace(rng) for _ in range(20)) if ns not in stored
    ]
    for namespace in absent:
        assert store.get(namespace, "k") is None, f"seed={seed}: get({namespace!r})"
