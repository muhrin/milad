# -*- coding: utf-8 -*-
"""
This module contains methods and tools to work with and invert the invariants that are similar
to those proposed in the following:

Bandeira, A. S., Blum-Smith, B., Kileel, J., Perry, A., Weed, J., & Wein, A. S. (2017).
Estimation under group actions: recovering orbits from invariants. ArXiv, (June).
http://arxiv.org/abs/1712.10163

We use an adapted version of their algorithm to recover moments from invariants.
"""
import math
from typing import Iterator, Tuple, Union

import numpy as np
from sympy.physics.quantum import cg

from . import invariants
from . import utils

# We use a lot of variable names that are convenient for mathematics but don't conform to the Google
# code style so just disable for this file
# pylint: disable=invalid-name


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

    def append(self, invariant: invariants.MomentInvariant):
        raise NotImplemented('Cannot append to invertible invariants')

    def invert(self, phi, moments_out):
        generator = InvariantsGenerator(self._l_le_n, self._n_minus_l_even)

        idx = 0
        # Degree 1, these are easy, all invariants are moments themselves
        for n in utils.inclusive(0, self._n_max, 2 if self._n_minus_l_even else 1):
            moments_out[n, 0, 0] = phi[idx]
            idx += 1

        # Degree 2 - Gram matrix time
        num_invs = generator.total_degree_2(self._n_max, 1)
        mtx_size = int(math.ceil(self._n_max / 2.)) if self._n_minus_l_even else self._n_max
        tri = np.zeros((mtx_size, mtx_size), dtype=complex)
        tri[np.triu_indices(mtx_size)] = phi[idx:idx + num_invs]
        ilower = np.tril_indices(mtx_size, -1)
        tri[ilower] = tri.T[ilower]

        vals = np.linalg.eigvals(tri)
        print(vals[vals < 0])

        L = np.linalg.cholesky(tri)  # Only uses lower triangular (plus diag)

        # Now, let's get a 3rd degree invariant that has all l = 1 so we can resolve the +/-I_3 degeneracy
        res = self.find(lambda inv: inv.weight == 3 and np.all(inv.terms_array[:, :, 1] == 1))

        for i, n in enumerate(utils.inclusive(1, self._n_max, 2 if self._n_minus_l_even else 1)):
            for m in (-1, 0, 1):
                moments_out[n, 1, m] = L[i, m + 1]

        inv_res = self[res[0]](moments_out)

        idx += num_invs

    def nl_pairs(self, n: Union[int, Tuple[int, int]], l: Union[int, Tuple[int, int]]) -> Iterator[Tuple[int, int]]:
        yield from utils.nl_pairs(n, l, l_le_n=self._l_le_n, n_minus_l_even=self._n_minus_l_even)


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

                    builder.add_term((-1)**k1 * complex(cg.CG(l2, k2, l3, k3, l1, -k1).doit()) / recip_prefactor,
                                     ((n1, l1, k1), (n2, l2, k2), (n3, l3, k3)))

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
        l_max = l_max or n_max
        done = set()

        invs = invariants.MomentInvariants()
        for pair_a in self.nl_pairs((1, n_max), (1, l_max)):
            for pair_b in self.nl_pairs((1, n_max), (1, l_max)):
                for pair_c in self.nl_pairs((1, n_max), (1, l_max)):
                    pairs = tuple(sorted([pair_a, pair_b, pair_c], key=lambda p: (p[1], p[0])))
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
