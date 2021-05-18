# -*- coding: utf-8 -*-
"""
This module contains methods and tools to work with and invert the invariants that are similar
to those proposed in the following:

Bandeira, A. S., Blum-Smith, B., Kileel, J., Perry, A., Weed, J., & Wein, A. S. (2017).
Estimation under group actions: recovering orbits from invariants. ArXiv, (June).
http://arxiv.org/abs/1712.10163

We use an adapted version of their algorithm to recover moments from invariants.
"""
import functools
import logging
import math
from typing import Iterator, Tuple, Union

import numpy as np
from sympy.physics.quantum import cg

from . import invariants
from . import utils

_LOGGER = logging.getLogger(__name__)

# We use a lot of variable names that are convenient for mathematics but don't conform to the Google
# code style so just disable for this file
# pylint: disable=invalid-name

Q = np.array([[-1 / 2**0.5, 0, 1j / 2**0.5], [0, 1, 0], [1 / 2**0.5, 0, 1j / 2**0.5]])
SQRT_THREE = 3**0.5


def cholesky(gram: np.ndarray) -> np.array:
    """Find Cholesky decomposition of the passed Gram matrix.  If this fails the algorithm
    will attempt to force it to be positive definite

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    # pylint: disable=invalid-name
    try:
        return np.linalg.cholesky(gram)
    except np.linalg.LinAlgError:
        pass

    B = (gram + gram.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    spacing = np.spacing(np.linalg.norm(gram))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(gram.shape[0])
    k = 1
    while True:
        try:
            return np.linalg.cholesky(A3)
        except np.linalg.LinAlgError:
            pass

        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1


class InvertibleInvariants(invariants.MomentInvariants):
    """A set of invertible invariants"""

    def __init__(
        self,
        degree_1: invariants.MomentInvariants,
        degree_2: invariants.MomentInvariants,
        degree_3: invariants.MomentInvariants,
        n_max: int,
        l_max: int,
        l_le_n: bool,
        n_minus_l_even: bool,
    ):
        super().__init__(*degree_1, *degree_2, *degree_3, are_real=False)
        self._degree_1 = degree_1
        self._degree_2 = degree_2
        self._degree_3 = degree_3
        self._l_le_n = l_le_n
        self._n_minus_l_even = n_minus_l_even
        self._n_max = n_max
        self._l_max = l_max

    def invert(self, phi: np.array, moments_out):
        used_invariants = set()
        moments_out.array.fill(float('nan'))

        used_invariants.update(self._invert_degree_1(phi, moments_out))
        used_invariants.update(self._invert_degree_2(phi, moments_out))
        used_invariants.update(self._invert_degree_3(phi, moments_out))

        _LOGGER.info('Used %i out of %i invariants during inversion', len(used_invariants), len(phi))
        return moments_out

    def nl_pairs(self, n: Union[int, Tuple[int, int]], l: Union[int, Tuple[int, int]]) -> Iterator[Tuple[int, int]]:
        yield from utils.nl_pairs(n, l, l_le_n=self._l_le_n, n_minus_l_even=self._n_minus_l_even)

    def _invert_degree_1(self, phi: np.array, moments_out) -> set:
        used_invariants = set()

        idx = 0
        # Degree 1, these are easy, all invariants are moments themselves
        for n in utils.inclusive(0, self._n_max, 2 if self._n_minus_l_even else 1):
            moments_out[n, 0, 0] = phi[idx]
            used_invariants.add(idx)
            idx += 1

        return used_invariants

    def _invert_degree_2(self, phi: np.array, moments_out) -> set:
        # pylint: disable=too-many-locals
        used_invariants = set()
        generator = InvariantsGenerator(self._l_le_n, self._n_minus_l_even)

        # Degree 2 - Gram matrix time
        start_idx = generator.total_degree_1(self._n_max)
        num_deg2 = generator.total_degree_2(self._n_max, 1)
        mtx_size = int(math.ceil(self._n_max / 2.)) if self._n_minus_l_even else self._n_max
        tri = np.zeros((mtx_size, mtx_size), dtype=complex)
        tri[np.triu_indices(mtx_size)] = phi[start_idx:start_idx + num_deg2]
        ilower = np.tril_indices(mtx_size, -1)
        tri[ilower] = tri.T[ilower]  # pylint: disable=unsubscriptable-object

        used_invariants.update(set(range(start_idx, start_idx + num_deg2)))

        # Let's do a Cholesky and extract a set of compatible X vectors
        L = cholesky(tri)[:, :3]  # Only uses lower triangular plus diag
        X1 = L @ Q.T

        for i, n in enumerate(utils.inclusive(1, self._n_max, 2 if self._n_minus_l_even else 1)):
            for m in (-1, 0, 1):
                moments_out[n, 1, m] = SQRT_THREE * X1[i, m + 1]

        # Now, let's get a 3rd degree invariant that has all l = 1 so we can resolve the +/-I_3 degeneracy
        res = np.array(self.find(lambda inv: inv.weight == 3 and np.all(inv.terms_array[:, :, 1] == 1)))
        # Get the index of a non-zero invariant that we can use to determine if we should use +Q or -Q
        i3_nonzero = np.argwhere(~np.isclose(phi[res], 0))[0][0]
        i3_nonzero = res[i3_nonzero]

        used_invariants.add(i3_nonzero)

        inv_res = self[i3_nonzero](moments_out)
        if np.isclose(inv_res / phi[i3_nonzero], -1.):
            # Need to use negative version of X1
            moments_out.array[:, 1, :] = -moments_out.array[:, 1, :]

        return used_invariants

    def _invert_degree_3(self, phi: np.array, moments_out) -> set:
        # pylint: disable=too-many-locals
        used_invariants = set()

        # Degree 3 - Now let's frequency march to recover the rest
        for l in utils.inclusive(2, self._l_max or self._n_max, 1):
            n_start = l if self._l_le_n else ((1 if utils.odd(l) else 2) if self._n_minus_l_even else 1)
            for n in utils.inclusive(n_start, self._n_max, 2 if self._n_minus_l_even else 1):
                # Get the correct invariants
                i3_idx = np.array(self.find(functools.partial(degree_3_with_unknown, n, l)))

                if len(i3_idx) < 2 * l + 1:
                    _LOGGER.warning("Don't have enough invariants to solve for n=%i, l=%i", n, l)

                # Create the matrices of the system to be solved
                coeffs = np.zeros((len(i3_idx), 2 * l + 1), dtype=complex)

                for i, inv_idx in enumerate(i3_idx):
                    invariant = self[inv_idx]
                    # Get just the first two terms (the third is the unknown)
                    for j, m in enumerate(utils.inclusive(-l, l)):
                        mask = invariant.terms_array[:, 2, 2] == m
                        if np.any(mask):
                            prefactors = invariant.prefactors[mask]
                            indices = invariant.terms_array[:, 0:2, :][mask]
                            # pylint: disable=protected-access
                            coeffs[i, j] = invariants._numpy_apply(prefactors, indices, moments_out)

                # The known invariants
                phis = phi[i3_idx]
                res = np.linalg.lstsq(coeffs, phis, rcond=None)

                used_invariants.update(set(i3_idx))

                for i, m in enumerate(utils.inclusive(-l, l)):
                    moments_out[n, l, m] = res[0][i]

        return used_invariants


def degree_3_with_unknown(n: int, l: int, inv: invariants.MomentInvariant):
    return inv.weight == 3 and np.all(inv.terms_array[:, 2, 0:2] == [n, l]) and np.all(inv.terms_array[:, 0:2, 1] < l)


class InvariantsGenerator:

    def __init__(self, l_le_n=True, n_minus_l_even=True):
        self._l_le_n = l_le_n
        self._n_minus_l_even = n_minus_l_even

    @staticmethod
    def delta(l1, l2, l3) -> bool:
        """Delta condition.  Returns True if |l2 - l3| <= l1 <= l2 + l3"""
        return abs(l2 - l3) <= l1 <= l2 + l3

    def total_degree_1(self, n_max: int) -> int:
        """Total number of degree 1 invariants up to n_max"""
        return (int(math.floor(n_max / 2.)) if self._n_minus_l_even else n_max) + 1

    def num_degree_2(self, n_max: int, l: int) -> int:
        if utils.odd(l):
            n = int(math.ceil(n_max / 2.)) if self._n_minus_l_even else n_max
        else:
            n = int(math.floor(n_max / 2.)) if self._n_minus_l_even else n_max

        return int(n * (n + 1) / 2)

    def total_degree_2(self, n_max: int, l_max=None) -> int:
        l_max = l_max or n_max
        return sum(self.num_degree_2(n_max, l) for l in utils.inclusive(1, l_max, 1))

    def inv_degree_2(self, n1: int, n2: int, l: int) -> invariants.MomentInvariant:
        """Generate a degree-2 invariant"""
        if self._l_le_n:
            err = None
            if l > n1:
                err = 'l must be <= n1, got n1={}, l={}'.format(n1, l)
            if l > n2:
                err = 'l must be <= n2, got n2={}, l={}'.format(n2, l)
            if err:
                raise ValueError(err)

        builder = invariants.InvariantBuilder(2)
        recip_prefactor = 2 * l + 1

        for k in utils.inclusive(-l, l, 1):
            builder.add_term((-1)**k / recip_prefactor, ((n1, l, k), (n2, l, -k)))

        return builder.build()

    def inv_degree_3(self, n1, l1, n2, l2, n3, l3):
        """Generate degree-2 invariant"""
        assert l1 <= n1, f'{l1} > {n1}'
        assert l2 <= n2, f'{l2} > {n2}'
        assert l3 <= n3, f'{l3} > {n3}'
        assert self.delta(l1, l2, l3)

        builder = invariants.InvariantBuilder(3)

        recip_prefactor = 2 * l1 + 1
        for k1 in utils.inclusive(-l1, l1):
            for k2 in utils.inclusive(-l2, l2):
                for k3 in utils.inclusive(-l3, l3):
                    if k1 + k2 + k3 != 0:
                        continue

                    prefactor = (-1)**k1 * complex(cg.CG(l2, k2, l3, k3, l1, -k1).doit()) / recip_prefactor
                    builder.add_term(prefactor, ((n1, l1, k1), (n2, l2, k2), (n3, l3, k3)))

        return builder.build()

    def generate_degree_1(self, n_max: int) -> invariants.MomentInvariants:
        """Generate first degree invariants up to the maximum n"""
        invs = invariants.MomentInvariants()
        for n in utils.inclusive(0, n_max, 2 if self._n_minus_l_even else 1):
            builder = invariants.InvariantBuilder(1)
            builder.add_term(1, [(n, 0, 0)])
            invs.append(builder.build())

        return invs

    def generate_degree_2(self, n_max: int, l_max=None) -> invariants.MomentInvariants:
        """Generate second degree invariants up to the maximum n, and optionally max l"""
        l_max = l_max or n_max
        assert n_max > 0
        assert l_max <= n_max

        invs = invariants.MomentInvariants()

        for n1 in utils.inclusive(1, n_max, 1):
            for n2 in utils.inclusive(n1, n_max, 1):
                l_max = min(l_max, n1) if self._l_le_n else l_max
                for l in utils.inclusive(1, l_max, 1):
                    if self._n_minus_l_even and (utils.odd(n1 - l) or utils.odd(n2 - l)):
                        continue

                    invs.append(self.inv_degree_2(n1, n2, l))

        return invs

    def generate_degree_3(self, n_max, l_max) -> invariants.MomentInvariants:
        """Generate third degree invariants up to maximum n, and optionally max l"""
        # pylint: disable=too-many-locals

        l_max = l_max or n_max
        done = set()

        invs = invariants.MomentInvariants()
        # We need one 3rd degree invariant at l=1 that includes three radial terms in order to resolve
        # the sign ambiguity on the unitary matrix from Cholesky decomposition
        if self._n_minus_l_even:
            invs.append(self.inv_degree_3(1, 1, 3, 1, 5, 1))
            done.add(((1, 1), (3, 1), (5, 1)))
        else:
            invs.append(self.inv_degree_3(1, 1, 2, 1, 3, 1))
            done.add(((1, 1), (2, 1), (3, 1)))

        for pair_c in self.nl_pairs((2 if self._l_le_n else 1, n_max), (2, l_max)):
            for pair_b in self.nl_pairs((1, n_max), (1, pair_c[1] - 1)):
                for pair_a in self.nl_pairs((1, n_max), (1, pair_b[1])):
                    pairs = tuple(sorted([pair_a, pair_b, pair_c], key=lambda p: (p[1], p[0])))
                    # pairs = pair_a, pair_b, pair_c
                    (n1, l1), (n2, l2), (n3, l3) = pairs

                    # Check delta criterion
                    if not self.delta(l1, l2, l3):
                        continue

                    # Check l <= n criterion
                    if self._l_le_n and (utils.odd(n1 - l1) or utils.odd(n2 - l2) or utils.odd(n3 - l3)):
                        continue

                    # Check for permutations that have already been done
                    if pairs in done:
                        continue

                    # Redundancy criteria
                    if utils.odd(l1) and pairs[0] == pairs[1] == pairs[2]:
                        continue
                    if utils.odd(l3) and pairs[0] == pairs[1] != pairs[2]:
                        continue

                    inv = self.inv_degree_3(n1, l1, n2, l2, n3, l3)
                    if inv == 0:
                        continue

                    invs.append(inv)
                    done.add(pairs)

        return invs

    def generate_all(self, n_max: int, l_max: int = None) -> InvertibleInvariants:
        """Generate all moments invariants using this scheme up to the max n and l"""
        invs = InvertibleInvariants(
            self.generate_degree_1(n_max=n_max), self.generate_degree_2(n_max=n_max, l_max=1),
            self.generate_degree_3(n_max=n_max, l_max=l_max), n_max, l_max, self._l_le_n, self._n_minus_l_even
        )

        return invs

    def nl_pairs(self, n: Union[int, Tuple[int, int]], l: Union[int, Tuple[int, int]]) -> Iterator[Tuple[int, int]]:
        yield from utils.nl_pairs(n, l, l_le_n=self._l_le_n, n_minus_l_even=self._n_minus_l_even)
