"""Compatibility imports for the small part of Diffrax's private typing API we use."""

from diffrax._custom_types import (  # pyright: ignore[reportPrivateUsage]
    Args,
    BoolScalarLike,
    DenseInfo,
    RealScalarLike,
    VF,
    Y,
)
from diffrax._term import WrapTerm  # pyright: ignore[reportPrivateUsage]
from diffrax._solver.srk import (  # pyright: ignore[reportPrivateUsage]
    AdditiveCoeffs,
    GeneralCoeffs,
)

__all__ = [
    "Args",
    "AdditiveCoeffs",
    "BoolScalarLike",
    "DenseInfo",
    "GeneralCoeffs",
    "RealScalarLike",
    "VF",
    "WrapTerm",
    "Y",
]
