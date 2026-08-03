"""Unified linear quantization backends."""

from .linear_backend import LinearBackendType, parse_linear_backend

__all__ = ["LinearBackendType", "parse_linear_backend"]
