import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from georax import SO
from georax._term import Pullback, PulledDiffusionTerm

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("operator", [False, True])
def test_diffusion_vf_prod_is_the_same_linear_map_at_each_stage(operator):
    geometry = SO(3)
    matrix = jnp.array([[0.3, -0.1], [0.2, 0.4], [-0.1, 0.5]])

    def vf(t, y, args):
        value = matrix * (1 + t + y[0, 1])
        return lx.MatrixLinearOperator(value) if operator else value

    term = diffrax.ControlTerm(vf, lambda t0, t1: jnp.zeros(2))
    pullback = Pullback(geometry, geometry.select_chart(2), jnp.eye(3))
    pulled = PulledDiffusionTerm(term, pullback)
    omega = jnp.array([0.3, -0.2, 0.1])
    dw = jnp.array([0.1, 0.2])

    def evaluated(omega):
        return pulled.prod(pulled.vf(0.2, omega, None), dw)

    expected = pulled.vf_prod(0.2, omega, None, dw)
    np.testing.assert_allclose(jax.jit(evaluated)(omega), expected, atol=1e-12)
    np.testing.assert_allclose(
        jax.jacfwd(evaluated)(omega),
        jax.jacfwd(lambda a: pulled.vf_prod(0.2, a, None, dw))(omega),
        atol=1e-12,
    )
    # Additive SRKs form linear combinations of vf results before contracting.
    vf0, vf1 = pulled.vf(0.1, omega, None), pulled.vf(0.3, omega, None)
    actual = pulled.prod(0.5 * (vf1 - vf0), dw)
    expected = 0.5 * (
        pulled.vf_prod(0.3, omega, None, dw) - pulled.vf_prod(0.1, omega, None, dw)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)
