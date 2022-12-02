# -*- coding: utf-8 -*-
import numpy as np

from milad import utils
from . import interfaces

__all__ = ("PowerSpectrum",)


class PowerSpectrum(interfaces.Invariants):
    """Calculate the power spectrum given a set of spherical harmonic based expansion coefficients"""

    supports_jacobian = False

    def __init__(self, mix_radials=True, radials_first=False):
        super().__init__()
        self._mix_radials = mix_radials
        self._radials_first = radials_first

    def evaluate(
        # pylint: disable=unused-argument
        self,
        state,
        *,
        get_jacobian=False
    ):
        # pylint: disable=invalid-name

        indices = state.indices
        invariants = []

        if self._radials_first:
            nmax = indices.n.max
            # Loop over radial functions first
            for n1 in utils.inclusive(nmax):
                n2_idx = utils.inclusive(n1, nmax, 2) if self._mix_radials else [n1]
                for n2 in n2_idx:
                    for l in indices.iter_l(min(n1, n2)):
                        invariants.append(
                            np.vdot(
                                state.array[n1, l, :].compressed(),
                                state.array[n2, l, :].compressed(),
                            ).real
                        )
        else:
            lmax = indices.l.max

            # Loops over angular frequencies first
            for l in utils.inclusive(lmax):
                for n1 in indices.iter_n(l):
                    n2_idx = (
                        indices.iter_n(l, (n1, None)) if self._mix_radials else [n1]
                    )
                    for n2 in n2_idx:
                        invariants.append(
                            np.vdot(
                                state.array[n1, l, :].compressed(),
                                state.array[n2, l, :].compressed(),
                            ).real
                        )

        return np.array(invariants)
