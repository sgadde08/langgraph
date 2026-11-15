"""Differential test of namespace matching against `InMemoryStore`.

`PostgresStore` flattens a namespace tuple into a dot-joined string and scopes reads
with `LIKE` and regular expressions, so the match semantics are re-implemented in
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
from langgraph.store.base import (
    GetOp,
    InvalidNamespaceError,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from langgraph.store.memory import InMemoryStore

from langgraph.store.postgres import PostgresStore
from tests.conftest import DEFAULT_URI

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
OPERATIONS_PER_SEED = 60


def _random_namespace(rng: random.Random) -> tuple[str, ...]:
    return tuple(rng.choice(LABELS) for _ in range(rng.randint(1, 3)))


def _namespaces(items: Any) -> list[tuple[str, ...]]:
    return sorted(tuple(namespace) for namespace in items)


@pytest.fixture
def store() -> Generator[PostgresStore, None, None]:
    with PostgresStore.from_conn_string(DEFAULT_URI) as store:
        store.setup()
        yield store


@pytest.mark.parametrize("seed", SEEDS)
def test_namespace_matching_agrees_with_in_memory_store(
    store: PostgresStore, seed: int
) -> None:
    rng = random.Random(seed)
    reference = InMemoryStore()
    stored = sorted({_random_namespace(rng) for _ in range(NAMESPACES_PER_SEED)})
    for position, namespace in enumerate(stored):
        reference.put(namespace, "k", {"position": position})
        store.put(namespace, "k", {"position": position}, index=False, ttl=None)

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


def _contents(store: Any) -> dict[tuple[tuple[str, ...], str], Any]:
    return {
        (tuple(item.namespace), item.key): item.value
        for item in store.search((), limit=10000)
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_writes_stay_inside_their_namespace(store: Any, seed: int) -> None:
    """Overwrites and deletes must only touch the namespace they address.

    A near-miss label (a sibling, a different case, a pattern metacharacter) must
    never resolve to another namespace's item, whether the write goes through
    `put`/`delete` or through `batch`.
    """
    rng = random.Random(seed)
    stored = sorted({_random_namespace(rng) for _ in range(NAMESPACES_PER_SEED)})
    keys = ["k", "k2"]
    expected: dict[tuple[tuple[str, ...], str], Any] = {}
    for position, namespace in enumerate(stored):
        for key in keys:
            value = {"position": position, "key": key}
            store.put(namespace, key, value, index=False, ttl=None)
            expected[(namespace, key)] = value
    assert _contents(store) == expected

    for step in range(OPERATIONS_PER_SEED):
        # Mostly address existing namespaces, but include near-misses of them.
        base = rng.choice(stored)
        if rng.random() < 0.5:
            namespace = base
        else:
            namespace = (
                (*base[:-1], rng.choice(LABELS))
                if rng.random() < 0.5
                else _random_namespace(rng)
            )
        key = rng.choice(keys)
        via_batch = rng.random() < 0.5
        if rng.random() < 0.5:
            value = {"step": step}
            if via_batch:
                store.batch([PutOp(namespace, key, value, index=False, ttl=None)])
            else:
                store.put(namespace, key, value, index=False, ttl=None)
            expected[(namespace, key)] = value
            action = "put"
        else:
            if via_batch:
                store.batch([PutOp(namespace, key, None)])
            else:
                store.delete(namespace, key)
            expected.pop((namespace, key), None)
            action = "delete"
        assert _contents(store) == expected, (
            f"seed={seed} step={step}: {action}({namespace!r}, {key!r})"
            f" via {'batch' if via_batch else 'method'}"
        )


DOTTED = ("tenant.bob",)  # would alias ("tenant", "bob") once dot-joined


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.get(DOTTED, "k"),
        lambda s: s.put(DOTTED, "k", {"v": "overwritten"}, index=False, ttl=None),
        lambda s: s.delete(DOTTED, "k"),
        lambda s: s.search(DOTTED),
        lambda s: s.list_namespaces(prefix=DOTTED),
        lambda s: s.list_namespaces(suffix=DOTTED),
        lambda s: s.batch([GetOp(DOTTED, "k")]),
        lambda s: s.batch([PutOp(DOTTED, "k", {"v": "overwritten"})]),
        lambda s: s.batch([PutOp(DOTTED, "k", None)]),
        lambda s: s.batch([SearchOp(DOTTED)]),
        lambda s: s.batch(
            [ListNamespacesOp(match_conditions=(MatchCondition("prefix", DOTTED),))]
        ),
        lambda s: s.batch(
            [ListNamespacesOp(match_conditions=(MatchCondition("suffix", DOTTED),))]
        ),
    ],
    ids=[
        "get",
        "put",
        "delete",
        "search",
        "list_prefix",
        "list_suffix",
        "batch_get",
        "batch_put",
        "batch_delete",
        "batch_search",
        "batch_list_prefix",
        "batch_list_suffix",
    ],
)
def test_dotted_label_cannot_address_another_namespace(
    store: Any, call: Callable[[Any], Any]
) -> None:
    """A label containing `.` is rejected on every entry point, changing nothing.

    Namespaces are stored dot-joined, so `("tenant.bob",)` would otherwise address
    the namespace `("tenant", "bob")`.
    """
    store.put(("tenant", "bob"), "k", {"v": "bob"}, index=False, ttl=None)
    before = _contents(store)
    with pytest.raises(InvalidNamespaceError):
        call(store)
    assert _contents(store) == before
