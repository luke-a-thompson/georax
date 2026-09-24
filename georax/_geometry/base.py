from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

Geometry = TypeVar("Geometry", bound="Manifold[Any]")
CoeffField = Callable[[Array], Array]
FrameCoords = Array
StateArray = Array


class LocalChart(eqx.Module, Generic[Geometry]):
    order: int = eqx.field(static=True)

    @abstractmethod
    def apply(self, x: Array, a: Array, geometry: Geometry) -> Array:
        """Apply the chosen local chart at x for coefficients a."""
        ...

    def inverse_differential(
        self, x: Array, a: Array, b: Array, geometry: Geometry
    ) -> Array:
        """Apply the inverse differential of ``a -> apply(x, a, geometry)``."""
        y, pushforward = jax.linearize(lambda aa: self.apply(x, aa, geometry), a)
        basis = jnp.eye(a.size, dtype=a.dtype)

        def apply_column(e: Array) -> Array:
            ambient_tangent = pushforward(e.reshape(a.shape))
            return geometry.trivialise(y, ambient_tangent).ravel()

        jac = jax.vmap(apply_column)(basis).T
        return jnp.linalg.solve(jac, b.ravel()).reshape(a.shape)


class Manifold(eqx.Module, Generic[Geometry]):
    """Minimal manifold geometry backend for retraction-based integrators.

    This class contains only geometric primitives and does not contain solver
    logic. It is sufficient for retraction-based manifold Runge--Kutta schemes
    such as the current commutator-free solvers.
    """

    _chart_class: ClassVar[type[LocalChart[Any]]]

    @property
    @abstractmethod
    def state_shape(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def coordinate_shape(self) -> tuple[int, ...]: ...

    @abstractmethod
    def trivialise(self, x: StateArray, v: StateArray) -> FrameCoords:
        """Convert ambient tangent vector v at state x to frame coordinates."""
        ...

    @abstractmethod
    def detrivialise(self, x: StateArray, a: FrameCoords) -> StateArray:
        """Convert frame coordinates a at state x to an ambient tangent vector."""
        ...

    @abstractmethod
    def frame_bracket(
        self, x: StateArray, a: FrameCoords, b: FrameCoords
    ) -> FrameCoords:
        """Bracket of constant frame-coordinate fields."""
        ...

    def check_state_shape(self, x: StateArray) -> None:
        if x.shape != self.state_shape:
            raise ValueError(
                f"{type(self).__name__} state must have shape {self.state_shape}; got {x.shape}."
            )

    def check_coordinate_shape(self, a: FrameCoords) -> None:
        if a.shape != self.coordinate_shape:
            raise ValueError(
                f"{type(self).__name__} coordinates must have shape {self.coordinate_shape}; got {a.shape}."
            )

    def apply_increment(
        self: Geometry, x: StateArray, a: FrameCoords, chart: LocalChart[Geometry]
    ) -> StateArray:
        """Apply frame coefficients at x using an explicitly selected chart."""
        self.check_state_shape(x)
        self.check_coordinate_shape(a)
        return chart.apply(x, a, self)

    def zero_coordinates(self, x: StateArray) -> FrameCoords:
        """Return the zero frame coordinates associated with state ``x``."""
        self.check_state_shape(x)
        return jnp.zeros(self.coordinate_shape, dtype=jnp.result_type(x))

    def select_chart(self: Geometry, required_order: int) -> LocalChart[Geometry]:
        """Return a local chart without changing this geometry."""
        return self._chart_class(required_order)

    def select_pullback_chart(
        self: Geometry, required_order: int
    ) -> LocalChart[Geometry]:
        """Choose coordinates for a solver that integrates a pulled-back equation."""
        return self.select_chart(required_order)


def covariant_derivative(
    geometry: Manifold[Any],
    a_fn: CoeffField,
    b_fn: CoeffField,
    x: Array,
) -> Array:
    a = a_fn(x)
    return jax.jvp(b_fn, (x,), (geometry.detrivialise(x, a),))[1]


def post_lie_bracket(
    geometry: Manifold[Any],
    a_fn: CoeffField,
    b_fn: CoeffField,
    x: Array,
) -> Array:
    a, nabla_b_a = jax.jvp(a_fn, (x,), (geometry.detrivialise(x, b_fn(x)),))
    b, nabla_a_b = jax.jvp(b_fn, (x,), (geometry.detrivialise(x, a),))
    return nabla_a_b - nabla_b_a + geometry.frame_bracket(x, a, b)
