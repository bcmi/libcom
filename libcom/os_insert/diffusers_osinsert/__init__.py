from __future__ import annotations

import importlib
from contextlib import contextmanager
import sys
import types
from pathlib import Path


def _load_patched_pipeline_source() -> str:
    src_path = Path(__file__).resolve().parent / "_patched_pipeline_flux_fill.py"
    return src_path.read_text(encoding="utf-8")


_ORIG_CALL = None
_PATCHED_CALL = None
_PATCH_ENABLED = False


def _load_patched_call():
    global _PATCHED_CALL
    if _PATCHED_CALL is not None:
        return _PATCHED_CALL

    patched_mod_name = "diffusers_osinsert._patched_pipeline_flux_fill"
    patched_mod = types.ModuleType(patched_mod_name)
    patched_mod.__file__ = str(Path(__file__).resolve().parent / "_patched_pipeline_flux_fill.py")
    # Make relative imports inside the patched file resolve against diffusers.*
    patched_mod.__package__ = "diffusers.pipelines.flux"

    code = _load_patched_pipeline_source()
    exec(compile(code, patched_mod.__file__, "exec"), patched_mod.__dict__)
    sys.modules[patched_mod_name] = patched_mod

    _PATCHED_CALL = patched_mod.FluxFillPipeline.__call__
    return _PATCHED_CALL


def enable_patch() -> None:
    """Enable the OSInsert FluxFillPipeline patch for the current Python process."""
    global _ORIG_CALL, _PATCH_ENABLED
    if _PATCH_ENABLED:
        return

    upstream_mod = importlib.import_module("diffusers.pipelines.flux.pipeline_flux_fill")
    if _ORIG_CALL is None:
        _ORIG_CALL = upstream_mod.FluxFillPipeline.__call__

    upstream_mod.FluxFillPipeline.__call__ = _load_patched_call()
    _PATCH_ENABLED = True


def disable_patch() -> None:
    """Disable the OSInsert FluxFillPipeline patch and restore upstream behavior."""
    global _PATCH_ENABLED
    if not _PATCH_ENABLED:
        return

    upstream_mod = importlib.import_module("diffusers.pipelines.flux.pipeline_flux_fill")
    if _ORIG_CALL is not None:
        upstream_mod.FluxFillPipeline.__call__ = _ORIG_CALL

    _PATCH_ENABLED = False


@contextmanager
def patch_context():
    """Temporarily enable the patch within a `with` block and restore afterwards."""
    enable_patch()
    try:
        yield
    finally:
        disable_patch()


# Re-export upstream classes for convenience.
from diffusers.pipelines.flux.pipeline_flux_fill import FluxFillPipeline  # noqa: E402
from diffusers import FluxPriorReduxPipeline  # noqa: E402
