"""Tests for GPU auto-select + batch-size defaults in the cross-encoder reranker.

These are pure unit tests — no model is loaded. We only exercise
``_resolve_device`` and ``_default_batch_size`` with monkeypatched inputs.
"""
from __future__ import annotations

import sys
import types

import pytest

from ext.services import cross_encoder_reranker as cer


# ---------------------------------------------------------------------------
# _resolve_device — explicit values
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_explicit(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_DEVICE", "cpu")
    assert cer._resolve_device() == "cpu"


def test_resolve_device_cuda_1_explicit(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_DEVICE", "cuda:1")
    assert cer._resolve_device() == "cuda:1"


def test_resolve_device_cuda_bare(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_DEVICE", "cuda")
    assert cer._resolve_device() == "cuda"


def test_resolve_device_case_insensitive(monkeypatch):
    # Uppercase should still resolve: we lowercase the env var before
    # comparing to "auto" / "cpu" etc.
    monkeypatch.setenv("RAG_RERANK_DEVICE", "CPU")
    assert cer._resolve_device() == "cpu"


# ---------------------------------------------------------------------------
# _resolve_device — auto path (default)
# ---------------------------------------------------------------------------


def _install_fake_torch(monkeypatch, *, cuda_available: bool):
    """Install a fake ``torch`` module with a controllable cuda.is_available."""
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_resolve_device_auto_returns_cuda0_when_available(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_DEVICE", "auto")
    _install_fake_torch(monkeypatch, cuda_available=True)
    assert cer._resolve_device() == "cuda:0"


def test_resolve_device_auto_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_DEVICE", "auto")
    _install_fake_torch(monkeypatch, cuda_available=False)
    assert cer._resolve_device() == "cpu"


def test_resolve_device_default_is_auto(monkeypatch):
    """No env var set → treated as auto."""
    monkeypatch.delenv("RAG_RERANK_DEVICE", raising=False)
    _install_fake_torch(monkeypatch, cuda_available=False)
    assert cer._resolve_device() == "cpu"


def test_resolve_device_auto_no_torch_module(monkeypatch):
    """If ``import torch`` fails entirely, auto must return ``cpu``."""
    monkeypatch.setenv("RAG_RERANK_DEVICE", "auto")
    # Setting to None forces ImportError on ``import torch``.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert cer._resolve_device() == "cpu"


# ---------------------------------------------------------------------------
# _default_batch_size
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, device: str | None):
        self.device = device


def test_default_batch_size_gpu_is_32(monkeypatch):
    monkeypatch.delenv("RAG_RERANK_BATCH_SIZE", raising=False)
    assert cer._default_batch_size(_FakeModel("cuda:0")) == 32


def test_default_batch_size_cpu_is_8(monkeypatch):
    monkeypatch.delenv("RAG_RERANK_BATCH_SIZE", raising=False)
    assert cer._default_batch_size(_FakeModel("cpu")) == 8


def test_default_batch_size_env_override_on_gpu(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_BATCH_SIZE", "64")
    assert cer._default_batch_size(_FakeModel("cuda:0")) == 64


def test_default_batch_size_env_override_on_cpu(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_BATCH_SIZE", "4")
    assert cer._default_batch_size(_FakeModel("cpu")) == 4


def test_default_batch_size_env_bogus_falls_back(monkeypatch):
    """A non-integer env var shouldn't crash; fall back to device default."""
    monkeypatch.setenv("RAG_RERANK_BATCH_SIZE", "not-a-number")
    assert cer._default_batch_size(_FakeModel("cuda:0")) == 32
    assert cer._default_batch_size(_FakeModel("cpu")) == 8


def test_default_batch_size_falls_back_to_target_device(monkeypatch):
    """Older sentence-transformers (<5) used ``_target_device``; still supported."""
    monkeypatch.delenv("RAG_RERANK_BATCH_SIZE", raising=False)

    class _OldModel:
        device = None
        _target_device = "cuda:0"

    assert cer._default_batch_size(_OldModel()) == 32


def test_default_batch_size_no_device_defaults_cpu(monkeypatch):
    """If neither .device nor ._target_device is set, assume CPU."""
    monkeypatch.delenv("RAG_RERANK_BATCH_SIZE", raising=False)

    class _DeviceFreeModel:
        pass

    assert cer._default_batch_size(_DeviceFreeModel()) == 8
