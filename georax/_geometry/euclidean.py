from __future__ import annotations

from typing import override

from diffrax._custom_types import RealScalarLike
import jax.numpy as jnp
from jaxtyping import Array

from .base import LocalFlow, Manifold, flow_order


class EuclideanFlow(LocalFlow):
    order: flow_order = "exact"
    inverse_order: flow_order = "exact"

    def forward(self, x: Array, a: Array, geometry: Euclidean) -> Array:
        return x + a

    def d_inverse(self, x: Array, y: Array, geometry: Euclidean) -> Array:
        return y


class Euclidean(Manifold):
    """R^n with the standard frame, addition retraction, and exact flow."""

    def __init__(self, *, flow: LocalFlow | None = None):
        object.__setattr__(self, "flow", EuclideanFlow() if flow is None else flow)

    @override
    def frame(self, x: Array) -> Array:
        return jnp.eye(x.shape[0], dtype=x.dtype)

    @override
    def to_frame(self, x: Array, v: Array) -> Array:
        del x
        return v

    @override
    def from_frame(self, x: Array, a: Array) -> Array:
        del x
        return a

    @override
    def retraction(self, x: Array, v: Array) -> Array:
        return x + v

    @override
    def select_flow_method(self, required_order: RealScalarLike) -> LocalFlow:
        del required_order
        flow: LocalFlow = EuclideanFlow()
        object.__setattr__(self, "flow", flow)
        return flow
