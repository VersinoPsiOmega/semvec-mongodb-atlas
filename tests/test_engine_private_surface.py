"""`MongoStateStore` inherits from the engine and skips its constructor.

*As a maintainer, I want a rename inside the engine's `MongoStateStore` to fail a test
here, so that the subclass which sets that class's private attributes by hand breaks
loudly instead of at the first `save()` in production.*

**Why this file exists.** `src/semvec_atlas/session_store.py` subclasses
`semvec.api.mongo_state_store.MongoStateStore` and deliberately does not call
`super().__init__()` — the upstream constructor resolves a client from the environment,
while this package already holds a tenant-scoped `Database`. It therefore assigns the five
private attributes the inherited methods read: `_collection`, `_event_store`, `_uri`,
`_db_name`, `_txn_supported`.

That is a contract against a **private** surface, and its own comment claimed two
safeguards: "the lineage tests and the version pin `<0.9`". Checked on 2026-09-20, both
halves were wrong. `test_upstream_lineage.py` pins four *public* identities — the guard
exception, the scope encoding, the rank-fusion probe, the PEM resolver — and none of these
five names; the only file in the repository that mentions `_txn_supported` or `_db_name`
is `session_store.py` itself. And the version cap has now been dropped, because capping at
`<0.9` would have made a `semvec` 0.9 uninstallable beside this package rather than
telling anyone why.

So the cap is replaced by the check it was standing in for. This is strictly the better
trade: a cap turns an incompatibility into an unsatisfiable resolution, which reads as a
packaging mistake; this turns it into a named failing test.

**Why the source and not an instance.** Constructing the upstream store needs a client and
a reachable MongoDB, which would make this guard `mongo`-marked and therefore absent from
exactly the quick run that should catch a bad upgrade. Reading the class source needs
neither, and a rename is what it detects.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from semvec.api.mongo_state_store import MongoStateStore as Engine

# The attributes `session_store.MongoStateStore.__init__` assigns in place of calling
# the upstream constructor. Each must still be something the upstream class reads.
PRIVATE_SURFACE = ("_collection", "_event_store", "_uri", "_db_name", "_txn_supported")

# The inherited methods this package relies on rather than reimplementing.
INHERITED_METHODS = ("save", "load", "load_full", "load_owner", "current_version", "delete")


@pytest.fixture(scope="module")
def engine_source() -> str:
    return inspect.getsource(Engine)


@pytest.mark.parametrize("attr", PRIVATE_SURFACE)
def test_the_engine_still_reads_the_attribute_we_set(attr: str, engine_source: str) -> None:
    """A rename upstream leaves our hand-set attribute dangling and silently unread."""
    assert f"self.{attr}" in engine_source, (
        f"`semvec.api.mongo_state_store.MongoStateStore` no longer uses `self.{attr}`. "
        f"`semvec_atlas.session_store.MongoStateStore.__init__` sets it by hand in place of "
        f"calling super().__init__(), so the value is now written and never read — every "
        f"inherited method would fall back to whatever the engine reads instead."
    )


@pytest.mark.parametrize("name", INHERITED_METHODS)
def test_the_inherited_method_is_still_there(name: str) -> None:
    """The subclass inherits these rather than implementing them."""
    assert callable(getattr(Engine, name, None)), (
        f"`MongoStateStore.{name}` is gone from the engine. `semvec_atlas` inherits it "
        f"instead of implementing it, so this is a removal, not a rename to work around."
    )


def test_our_subclass_really_does_skip_the_upstream_constructor() -> None:
    """Non-vacuity for everything above.

    All of it matters only while the subclass keeps setting those attributes itself. If it
    ever starts calling `super().__init__()`, the private surface stops being our problem
    and this file should go rather than quietly pass.
    """
    from semvec_atlas.session_store import MongoStateStore as Ours

    own_init = inspect.getsource(Ours.__init__)

    # Parsed, not grepped. The method's own comment contains the words
    # "Deliberately no super().__init__()", so a substring search finds the
    # explanation and reports it as the call it is explaining — which is what the
    # first version of this test did.
    body = ast.parse(textwrap.dedent(own_init)).body[0]
    calls_super = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__init__"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        for node in ast.walk(body)
    )
    assert not calls_super, (
        "the subclass now calls super().__init__(), so it no longer sets the engine's "
        "private attributes by hand — delete this guard instead of leaving it green"
    )
    for attr in PRIVATE_SURFACE:
        assert f"self.{attr}" in own_init, (
            f"the subclass no longer sets `{attr}`; update PRIVATE_SURFACE so this guard "
            f"keeps describing what the code actually does"
        )
