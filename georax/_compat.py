"""Compatibility imports for the small part of Diffrax's private typing API we use."""

from diffrax._custom_types import (  # pyright: ignore[reportPrivateUsage]
    VF,
    Args,
    BoolScalarLike,
    Control,
    DenseInfo,
    RealScalarLike,
    Y,
)
from diffrax._solver.srk import (  # pyright: ignore[reportPrivateUsage]
    AdditiveCoeffs,
    GeneralCoeffs,
)
from diffrax._term import WrapTerm  # pyright: ignore[reportPrivateUsage]

__all__ = [
    "Args",
    "AdditiveCoeffs",
    "BoolScalarLike",
    "Control",
    "DenseInfo",
    "GeneralCoeffs",
    "RealScalarLike",
    "VF",
    "WrapTerm",
    "Y",
]
