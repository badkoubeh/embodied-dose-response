"""The stub tier exists to pin contracts, so the contracts get checked."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import subprocess
import sys
import typing

import pytest

import edr

STUB_PACKAGES = ["edr.data", "edr.runner", "edr.metrics", "edr.analysis"]


def _modules(package_names):
    for name in package_names:
        pkg = importlib.import_module(name)
        yield pkg
        for info in pkgutil.iter_modules(pkg.__path__):
            yield importlib.import_module(f"{name}.{info.name}")


ALL_MODULES = list(_modules(["edr", "edr.perturb", *STUB_PACKAGES]))


@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda m: m.__name__)
def test_all_annotations_resolve(module):
    """Catches unresolvable forward refs, typo'd type names, and missing imports.

    The entire value of a stub is its signature, so an unchecked signature is a
    stub that pins nothing.
    """
    for _, obj in inspect.getmembers(module):
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == module.__name__:
            typing.get_type_hints(obj)


@pytest.mark.parametrize("module", list(_modules(STUB_PACKAGES)), ids=lambda m: m.__name__)
def test_stub_callables_raise_not_implemented(module):
    """A stub must fail with NotImplementedError, not TypeError -- a TypeError
    means the signature itself is wrong."""
    for name, obj in vars(module).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        # Dataclasses in these modules are real data containers (Inference,
        # CellSpec), not stubs -- they are expected to construct.
        if dataclasses.is_dataclass(obj):
            continue
        params = inspect.signature(obj).parameters
        args = [None] * sum(
            1
            for p in params.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.name != "self"
        )
        with pytest.raises(NotImplementedError):
            obj(*args)


def test_analysis_plane_imports_without_torch():
    """PLAN.md 2.1's plane split, as an executable invariant rather than a
    convention that erodes on the first convenient import.

    A scoring module that reaches into `edr.runner` would drag torch into the
    analysis plane and quietly break laptop installs.
    """
    code = (
        "import sys;"
        "import edr.schema, edr.seeding, edr.perturb, edr.metrics, edr.analysis;"
        "assert 'torch' not in sys.modules, 'torch leaked into the analysis plane';"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_root_package_stays_import_light():
    assert not hasattr(edr, "np")
    assert edr.__version__
