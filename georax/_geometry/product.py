from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from .base import FrameCoords, LocalChart, Manifold, StateArray
from .euclidean import Euclidean
from .torus import Torus

__all__ = ["Product", "TangentTorus"]


class ProductChart(LocalChart["Product"]):
    charts: tuple[LocalChart[Any], ...]

    def apply(self, x: Array, a: Array, geometry: Product) -> Array:
        return geometry.pack_state(
            *(
                factor.apply_increment(xi, ai, chart)
                for factor, chart, xi, ai in zip(
                    geometry.factors,
                    self.charts,
                    geometry.unpack_state(x),
                    geometry.unpack_coordinates(a),
                    strict=True,
                )
            )
        )

    def inverse_differential(
        self, x: Array, a: Array, b: Array, geometry: Product
    ) -> Array:
        return geometry.pack_coordinates(
            *(
                chart.inverse_differential(xi, ai, bi, factor)
                for factor, chart, xi, ai, bi in zip(
                    geometry.factors,
                    self.charts,
                    geometry.unpack_state(x),
                    geometry.unpack_coordinates(a),
                    geometry.unpack_coordinates(b),
                    strict=True,
                )
            )
        )


class Product(Manifold["Product"]):
    """Cartesian product with flat array states and flat frame coordinates.

    Factors keep their own state and coordinate shapes. Packing concatenates
    their flattened arrays in factor order; unpacking restores these shapes.
    Products can be nested. All factors must declare fixed shapes.
    """

    _chart_class = ProductChart
    factors: tuple[Manifold[Any], ...]

    def __init__(self, *factors: Manifold[Any]) -> None:
        if not factors:
            raise ValueError("Product requires at least one factor.")
        for factor in factors:
            if not isinstance(factor, Manifold):
                raise TypeError("Product factors must be Manifold instances.")
            # Fail at construction if a factor has no fixed shape.
            _ = factor.state_shape, factor.coordinate_shape
        self.factors = factors

    @property
    def state_shape(self) -> tuple[int]:
        return (sum(math.prod(f.state_shape) for f in self.factors),)

    @property
    def coordinate_shape(self) -> tuple[int]:
        return (sum(math.prod(f.coordinate_shape) for f in self.factors),)

    @staticmethod
    def _pack(arrays: tuple[Array, ...], shapes: tuple[tuple[int, ...], ...]) -> Array:
        if len(arrays) != len(shapes):
            raise ValueError(
                f"Expected {len(shapes)} factor arrays; got {len(arrays)}."
            )
        arrays = tuple(jnp.asarray(a) for a in arrays)
        for a, shape in zip(arrays, shapes, strict=True):
            if a.shape != shape:
                raise ValueError(
                    f"Factor array must have shape {shape}; got {a.shape}."
                )
        return jnp.concatenate([a.reshape(-1) for a in arrays])

    @staticmethod
    def _unpack(array: Array, shapes: tuple[tuple[int, ...], ...]) -> tuple[Array, ...]:
        # Leading axes (e.g. saved observation times) are retained.
        size = sum(math.prod(shape) for shape in shapes)
        if array.ndim == 0 or array.shape[-1] != size:
            raise ValueError(
                f"Packed array must have final dimension {size}; got {array.shape}."
            )
        parts = []
        start = 0
        for shape in shapes:
            end = start + math.prod(shape)
            parts.append(array[..., start:end].reshape(array.shape[:-1] + shape))
            start = end
        return tuple(parts)

    def pack_state(self, *states: Array) -> StateArray:
        """Pack one state array per factor into a flat solver state."""
        return self._pack(states, tuple(f.state_shape for f in self.factors))

    def unpack_state(self, x: StateArray) -> tuple[Array, ...]:
        """Restore factor states, retaining any leading observation axes."""
        return self._unpack(x, tuple(f.state_shape for f in self.factors))

    def pack_coordinates(self, *coordinates: Array) -> FrameCoords:
        """Pack one frame-coordinate array per factor."""
        return self._pack(coordinates, tuple(f.coordinate_shape for f in self.factors))

    def unpack_coordinates(self, a: FrameCoords) -> tuple[Array, ...]:
        """Restore factor coordinates, retaining any leading observation axes."""
        return self._unpack(a, tuple(f.coordinate_shape for f in self.factors))

    def trivialise(self, x: StateArray, v: StateArray) -> FrameCoords:
        self.check_state_shape(x)
        self.check_state_shape(v)
        return self.pack_coordinates(
            *(
                f.trivialise(xi, vi)
                for f, xi, vi in zip(
                    self.factors,
                    self.unpack_state(x),
                    self.unpack_state(v),
                    strict=True,
                )
            )
        )

    def detrivialise(self, x: StateArray, a: FrameCoords) -> StateArray:
        self.check_state_shape(x)
        self.check_coordinate_shape(a)
        return self.pack_state(
            *(
                f.detrivialise(xi, ai)
                for f, xi, ai in zip(
                    self.factors,
                    self.unpack_state(x),
                    self.unpack_coordinates(a),
                    strict=True,
                )
            )
        )

    def frame_bracket(
        self, x: StateArray, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        self.check_state_shape(x)
        self.check_coordinate_shape(a)
        self.check_coordinate_shape(b)
        return self.pack_coordinates(
            *(
                f.frame_bracket(xi, ai, bi)
                for f, xi, ai, bi in zip(
                    self.factors,
                    self.unpack_state(x),
                    self.unpack_coordinates(a),
                    self.unpack_coordinates(b),
                    strict=True,
                )
            )
        )

    def select_chart(self, required_order: int) -> ProductChart:
        return ProductChart(
            required_order, tuple(f.select_chart(required_order) for f in self.factors)
        )

    def select_pullback_chart(self, required_order: int) -> ProductChart:
        return ProductChart(
            required_order,
            tuple(f.select_pullback_chart(required_order) for f in self.factors),
        )


class TangentTorus(Product):
    """Torus(n) x R^n, packed as [theta, omega]; only theta is wrapped."""

    def __init__(self, n: int) -> None:
        super().__init__(Torus(n), Euclidean(n))
