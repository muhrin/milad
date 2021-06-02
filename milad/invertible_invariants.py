# -*- coding: utf-8 -*-
"""
This module contains methods and tools to work with and invert the invariants that are similar
to those proposed in the following:

Bandeira, A. S., Blum-Smith, B., Kileel, J., Perry, A., Weed, J., & Wein, A. S. (2017).
Estimation under group actions: recovering orbits from invariants. ArXiv, (June).
http://arxiv.org/abs/1712.10163

We use an adapted version of their algorithm to recover moments from invariants.
"""
import collections
import functools
import logging
import math
from typing import Tuple

import numpy as np
from sympy.physics.quantum import cg

from . import invariants
from . import utils
from . import mathutil
from . import sph

_LOGGER = logging.getLogger(__name__)

# We use a lot of variable names that are convenient for mathematics but don't conform to the Google
# code style so just disable for this file
# pylint: disable=invalid-name
SQRT_TWO = 2**0.5


def q_matrix(l: int) -> np.ndarray:
    """Create a Q matrix that transforms real spherical harmonics to complex ones"""
    size = 2 * l + 1
    Q_ = np.zeros((size, size), dtype=complex)
    Q_[0, 0] = SQRT_TWO
    for m in utils.inclusive(1, l):
        Q_[m, -m] = 1
        Q_[m, m] = (-1)**m
        Q_[-m, m] = 1j
        Q_[-m, -m] = 1j

    return Q_ / SQRT_TWO


class InvertibleInvariants(invariants.MomentInvariants):
    """A set of invertible invariants"""

    def __init__(
        self, degree_1: invariants.MomentInvariants, degree_2: np.ma.masked_array,
        degree_3: invariants.MomentInvariants, index_traits: sph.IndexTraits
    ):
        super().__init__(*degree_1, *degree_2.compressed(), *degree_3, are_real=False)
        self._degree_1 = degree_1
        self._degree_2 = degree_2
        self._degree_3 = degree_3
        self._index_traits = index_traits

    def invert(self, phi: np.array, moments_out):
        l_max = self._index_traits.l[1]

        used_invariants = set()
        moments_out.array.fill(float('nan'))

        used_invariants.update(self._invert_degree_1(phi, moments_out))

        # Let's go up in l until we find an I2 we can solve for
        l = 1
        while l < l_max:
            gram, phi_indices = self.gram_matrix(phi, l=l)
            vectors, rank = self.get_vectors_from_gram(gram, l)
            self._place_vectors(moments_out, vectors, l)
            used_invariants.update(phi_indices)

            if rank > 0:
                # Now, let's get a 3rd degree invariant that have l1=l2=l3 so we can resolve the +/-I_3 degeneracy
                try:
                    sign, inv_idx = self._determine_sign(phi, moments_out, l)
                except RuntimeError:
                    _LOGGER.warning('Unable to determine sign for I_2, l=%i', l)
                else:
                    moments_out.array[:, l, :] = sign * moments_out.array[:, l, :]
                    used_invariants.add(inv_idx)

                break

            l += 1

        l += 1

        # Now we can continue using I3 only
        used_invariants.update(self._invert_degree_3(phi, l, moments_out))

        _LOGGER.info('Used %i out of %i invariants during inversion', len(used_invariants), len(phi))
        return moments_out

    def gram_matrix(self, phi: np.ndarray, l: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the Gram matrix for the corresponding angular frequency, l"""
        # Let's get the indices for the degree-2 invariants
        num_deg_1 = len(self._degree_1)
        indices = np.ma.masked_array(np.empty(self._degree_2.shape, dtype=int), mask=self._degree_2.mask)
        valid_invs = np.argwhere(self._degree_2 != None)  # pylint: disable=singleton-comparison
        indices[valid_invs[:, 0], valid_invs[:, 1], valid_invs[:, 2]] = (np.arange(0, len(valid_invs)) + num_deg_1)

        gram_size = sum(1 for _ in self._index_traits.iter_n(l))
        gram = np.zeros((gram_size, gram_size), dtype=complex)
        phi_indices = indices[l, :, :].compressed()
        gram[np.triu_indices(gram_size)] = phi[phi_indices]
        ilower = np.tril_indices(gram_size, -1)
        gram[ilower] = gram.T[ilower]  # pylint: disable=unsubscriptable-object

        return gram, phi_indices

    # @staticmethod
    # def get_vectors_from_gram(gram: np.array, l: int) -> Tuple[int, np.array]:
    #     """Given a Gram matrix that that is interpreted as an the SxS (where there are S radial functions) inner
    #     product space of the vectors at a particular angular frequency, l, this will solve for a set of possible
    #     vectors up to isomorphism.
    #     The rank of the Gram matrix and an array of the vectors are returned.
    #     """
    #     X1 = np.zeros((gram.shape[0], 2 * l + 1), dtype=complex)
    #     rank = np.linalg.matrix_rank(gram)
    #
    #     # Check if we have anything to decompose, if not just save ourselves the trouble and return now
    #     if not gram.nonzero() or rank == 0:
    #         return rank, X1
    #
    #     # Let's do a Cholesky and extract a set of compatible X vectors
    #     L = cholesky(gram)
    #     L = L[:, :rank]  # Collect entries up to rank
    #
    #     bound = int(math.floor(rank / 2))
    #     m_values = tuple(m for m in utils.inclusive(-bound, bound, 1) if not (mathutil.even(rank) and m == 0))
    #
    #     # Get a rotation matrix that makes the vectors respect the conjugate symmetry
    #     Q = q_matrix(l)[m_values, :][:, m_values]
    #     X1[:, m_values] = L @ Q
    #
    #     return rank, X1

    @staticmethod
    def get_vectors_from_gram(gram: np.array, l: int) -> Tuple[np.array, int]:
        """Given a Gram matrix that that is interpreted as an the SxS (where there are S radial functions) inner product
         pace of the vectors at a particular angular frequency, l, this will solve for a set of possible vectors up to
         isomorphism.
        The rank of the Gram matrix and an array of the vectors are returned.
        """
        vectors = np.zeros((gram.shape[0], 2 * l + 1), dtype=complex)

        u, s, _vh = np.linalg.svd(gram, hermitian=True)
        rank = int(np.sum(~np.isclose(s, 0.)))

        # Check if we have anything to decompose, if not just save ourselves the trouble and return now
        if rank == 0:
            return vectors, rank

        L = u[:, :rank] * s[:rank]**0.5

        bound = int(math.floor(rank / 2))
        m_values = tuple(m for m in utils.inclusive(-bound, bound, 1) if not (mathutil.even(rank) and m == 0))

        # Get a rotation matrix that makes the vectors respect the conjugate symmetry
        Q = q_matrix(l)[m_values, :][:, m_values]
        # Perform the rotation and multiply by the Clebsch-Gordan coefficient
        vectors[:, m_values] = (2 * l + 1)**0.25 * L @ Q

        return vectors, rank

    def _invert_degree_1(self, phi: np.array, moments_out) -> set:
        """Get the degree 1 invariants.  There is no inversion to be done here, we just copy over all the values
        corresponding to c_n0^0 which are all rotation invariants"""
        used_invariants = set()

        # Degree 1, these are easy, all invariants are moments themselves
        indices = tuple(self._index_traits.iter_n(l=0))
        num = len(indices)
        moments_out.array[indices, 0, 0] = phi[:num]
        used_invariants.update(range(num))

        return used_invariants

    def _place_vectors(self, moments, vectors, l: int):
        m_range = tuple(utils.inclusive(-l, l))
        for i, n in enumerate(self._index_traits.iter_n(l)):
            moments.array[n, l, m_range] = vectors[i, m_range]
        return moments

    def _determine_sign(self, phi: np.ndarray, moments_out, l: int):
        # Find I3 that have l1=l2=l3
        res = np.array(self.find(lambda inv: inv.weight == 3 and np.all(inv.terms_array[:, :, 1] == l)))
        # Get the index of a non-zero invariant that we can use to determine if we should use +Q or -Q
        try:
            # Find one that is non-zero
            i3_nonzero = np.argwhere(~np.isclose(phi[res], 0))[0][0]
        except IndexError:
            raise RuntimeError(f'Unable to determine sign for I_2, l={l}')
        else:
            i3_nonzero = res[i3_nonzero]
            inv_res = self[i3_nonzero](moments_out)
            sign = inv_res / phi[i3_nonzero]
            if np.isclose(sign, 1):
                return 1, i3_nonzero
            if np.isclose(sign, -1.):
                return -1, i3_nonzero

            raise RuntimeError(f'The passed moments do not match I3 invariant (idx={i3_nonzero})')

    def _invert_degree_3(self, phi: np.array, l_start: int, moments_out) -> set:
        # pylint: disable=too-many-locals
        used_invariants = set()

        # Degree 3 - Now let's frequency march to recover the rest
        for l3 in utils.inclusive(l_start, self._index_traits.l[1], 1):
            vectors, used = self._invert_deg_3_l(phi, moments_out, l3)
            self._place_vectors(moments_out, vectors, l3)
            used_invariants.update(used)

        return used_invariants

    def _invert_deg_3_l(self, phi: np.array, moments, l3: int):
        # pylint: disable=too-many-locals
        used_invariants = set()

        n_values = tuple(self._index_traits.iter_n(l3))
        m_values = tuple(self._index_traits.iter_m(l3))
        vectors = np.empty((len(n_values), 2 * l3 + 1), dtype=moments.dtype)

        for n3_, n3 in enumerate(self._index_traits.iter_n(l3)):
            # Get the correct invariants
            i3_idx = np.array(self.find(functools.partial(degree_3_with_unknown, n3, l3)))

            if len(i3_idx) < 2 * l3 + 1:
                _LOGGER.warning("Don't have enough invariants to solve for n=%i, l=%i", n3, l3)

            # Create the matrices of the system to be solved
            coeffs = np.zeros((len(i3_idx), 2 * l3 + 1), dtype=complex)

            for i, inv_idx in enumerate(i3_idx):
                invariant = self[inv_idx]
                # Get just the first two terms (the third is the unknown)
                for j, m in enumerate(utils.inclusive(-l3, l3)):
                    mask = invariant.terms_array[:, 2, 2] == m
                    if np.any(mask):
                        prefactors = invariant.prefactors[mask]
                        indices = invariant.terms_array[:, 0:2, :][mask]
                        # pylint: disable=protected-access
                        coeffs[i, j] = invariants._numpy_apply(prefactors, indices, moments)

            # The known invariants
            phis = phi[i3_idx]
            res = np.linalg.lstsq(coeffs, phis, rcond=None)
            if res[2] < (2 * l3 + 1):
                _LOGGER.warning('Rank deficiency (%i < %i) in n=%i, l=%i', res[2], (2 * l3 + 1), n3, l3)

            used_invariants.update(set(i3_idx))

            vectors[n3_, m_values] = res[0]

        return vectors, used_invariants


def degree_3_with_unknown(n: int, l: int, inv: invariants.MomentInvariant):
    return inv.weight == 3 and np.all(inv.terms_array[:, 2, 0:2] == [n, l]) and np.all(inv.terms_array[:, 0:2, 1] < l)


class InvariantsGenerator:

    @staticmethod
    def delta(l1, l2, l3) -> bool:
        """Delta condition.  Returns True if |l2 - l3| <= l1 <= l2 + l3"""
        return abs(l2 - l3) <= l1 <= l2 + l3

    @staticmethod
    def total_degree_1(index_traits: sph.IndexTraits) -> int:
        """Total number of degree 1 invariants up to n_max"""
        return sum(1 for _ in index_traits.iter_n(0))

    @staticmethod
    def num_degree_2(index_traits: sph.IndexTraits, l: int) -> int:
        n = sum(1 for _ in index_traits.iter_n(l))
        return int(n * (n + 1) / 2)

    @classmethod
    def total_degree_2(cls, index_traits: sph.IndexTraits) -> int:
        l_max = index_traits.l[1]
        return sum(cls.num_degree_2(index_traits, l) for l in utils.inclusive(1, l_max, 1))

    @staticmethod
    def inv_degree_2(n1: int, n2: int, l: int) -> invariants.MomentInvariant:
        """Generate a degree-2 invariant
        """
        builder = invariants.InvariantBuilder(2)
        # This comes from the Clebsch-Gordan coefficient: <l,l,m,-m|0,0> = (-1)^(l-m)/sqrt(2l + 1)
        recip_prefactor = (2 * l + 1)**0.5

        for m in utils.inclusive(-l, l, 1):
            builder.add_term((-1)**(l - m) / recip_prefactor, ((n1, l, m), (n2, l, -m)))

        return builder.build()

    @classmethod
    def inv_degree_3(cls, n1, l1, n2, l2, n3, l3):
        """Generate degree-3 invariant"""
        assert l1 <= n1, f'{l1} > {n1}'
        assert l2 <= n2, f'{l2} > {n2}'
        assert l3 <= n3, f'{l3} > {n3}'
        assert cls.delta(l1, l2, l3)

        builder = invariants.InvariantBuilder(3)

        recip_prefactor = (2 * l1 + 1)**0.5
        for k1 in utils.inclusive(-l1, l1):
            for k2 in utils.inclusive(-l2, l2):
                for k3 in utils.inclusive(-l3, l3):
                    if k1 + k2 + k3 != 0:
                        continue

                    prefactor = (-1)**k1 * complex(cg.CG(l2, k2, l3, k3, l1, -k1).doit()) / recip_prefactor
                    builder.add_term(prefactor, ((n1, l1, k1), (n2, l2, k2), (n3, l3, k3)))

        return builder.build()

    @staticmethod
    def generate_degree_1(index_traits: sph.IndexTraits) -> invariants.MomentInvariants:
        """Generate first degree invariants up to the maximum n"""
        invs = invariants.MomentInvariants()
        for n in index_traits.iter_n(0):
            builder = invariants.InvariantBuilder(1)
            builder.add_term(1, [(n, 0, 0)])
            invs.append(builder.build())

        return invs

    @classmethod
    def generate_degree_2(cls, index_traits: sph.IndexTraits) -> np.ma.masked_array:
        """Generate second degree invariants up to the maximum n, and optionally max l"""
        N = index_traits.N
        L = index_traits.L

        invs_array = np.empty((L, N, N), dtype=object)
        invs_array.fill(None)

        for l in utils.inclusive(1, index_traits.l.max, 1):
            for n1 in index_traits.iter_n(l):
                for n2 in index_traits.iter_n(l, n_spec=(n1, None)):
                    invs_array[l, n1, n2] = cls.inv_degree_2(n1, n2, l)

        return np.ma.masked_array(invs_array, invs_array == None)  # pylint: disable=singleton-comparison

    @classmethod
    def generate_degree_3(cls, index_traits: sph.IndexTraits) -> invariants.MomentInvariants:
        """Generate third degree invariants up to maximum n, and optionally max l"""
        # pylint: disable=too-many-locals

        l_max = index_traits.l[1]
        done = set()

        invs = invariants.MomentInvariants()

        # We need one 3rd degree invariant at l1=l2=l3 that includes three radial terms in order to resolve
        # the sign ambiguity on the unitary matrix from Cholesky decomposition
        for l in utils.inclusive(1, l_max):
            for n1 in index_traits.iter_n(l):
                for n2 in index_traits.iter_n(l, n_spec=(n1, None)):
                    for n3 in index_traits.iter_n(l, n_spec=(n2, None)):
                        if degree_3_is_zero((n1, l), (n2, l), (n3, l)):
                            continue

                        invs.append(cls.inv_degree_3(n1, l, n2, l, n3, l))
                        done.add(((n1, l), (n2, l), (n3, l)))

        for pair_c in index_traits.iter_nl(n_spec=(2, None), l_spec=(2, None)):
            for pair_b in index_traits.iter_nl(l_spec=(None, pair_c[1] - 1)):
                for pair_a in index_traits.iter_nl(l_spec=(None, pair_b[1])):
                    pairs = tuple(sorted([pair_a, pair_b, pair_c], key=lambda p: (p[1], p[0])))
                    # pairs = pair_a, pair_b, pair_c
                    (n1, l1), (n2, l2), (n3, l3) = pairs

                    # Check delta criterion
                    if not cls.delta(l1, l2, l3):
                        continue

                    # Check for permutations that have already been done
                    if pairs in done:
                        continue

                    # Check for zero invariants
                    if degree_3_is_zero(*pairs):
                        continue

                    inv = cls.inv_degree_3(n1, l1, n2, l2, n3, l3)
                    invs.append(inv)
                    done.add(pairs)

        return invs

    @classmethod
    def generate_all(cls, index_traits: sph.IndexTraits) -> InvertibleInvariants:
        """Generate all moments invariants using this scheme up to the max n and l"""
        invs = InvertibleInvariants(
            cls.generate_degree_1(index_traits), cls.generate_degree_2(index_traits),
            cls.generate_degree_3(index_traits), index_traits
        )

        return invs


def degree_3_is_zero(pair1: Tuple, pair2: Tuple, pair3: Tuple) -> bool:
    """Check if a set a degree-3 invariant is identically zero"""
    if mathutil.odd(pair1[1]) and pair1 == pair2 == pair3:
        # All same
        return True

    counts = collections.defaultdict(int)
    counts[pair1] += 1
    counts[pair2] += 1
    counts[pair3] += 1

    if mathutil.odd(pair3[1]) and pair1 == pair2 and pair1 != pair3:
        return True
    if mathutil.odd(pair2[1]) and pair1 == pair3 and pair1 != pair2:
        return True
    if mathutil.odd(pair1[1]) and pair2 == pair3 and pair2 != pair1:
        return True

    return False
