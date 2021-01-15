# -*- coding: utf-8 -*-
"""Module that is concerned with the calculation of moment invariants"""
import collections
import functools
import operator
import pathlib
from typing import Sequence, Union, List, Set, Tuple, Dict, Iterator

import numba
import numpy as np

from . import base_moments
from . import functions
from . import geometric

__all__ = 'MomentInvariant', 'read_invariants', 'RES_DIR'

# The resources directory
RES_DIR = pathlib.Path(__file__).parent / 'res'
GEOMETRIC_INVARIANTS = RES_DIR / 'rot3dinvs8mat.txt'
COMPLEX_INVARIANTS = RES_DIR / 'cmfs7indep_0.txt'


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)


class MomentInvariant:
    """Class storing moment invariants.

    The invariants consist of a sum of terms where each term has a prefactor multiplied by a product
    of moments labelled by three indices e.g. p * c_20^0 * c_20^0
    """

    def __init__(self, weight):
        self._weight = weight
        self._terms = []
        self._max_order = -1
        self._constant = 0

        self._farray = None  # The prefactors array
        self._indarray = None  # The index array
        self._norm_power = None
        self._derivatives = None  # A cache for the derivatives

    def __str__(self):
        sum_parts = []
        for prefactor, product in self._terms:
            powers = self._collect_powers(product)
            product_parts = [str(prefactor)]
            product_parts.extend(
                'm{},{},{}^{}'.format(indices[0], indices[1], indices[2], power) for indices, power in powers.items()
            )

            string = ' '.join(product_parts)
            sum_parts.append(string)

        return ' + '.join(sum_parts)

    @property
    def weight(self) -> int:
        """The number of terms in each product of the invariant.  This gives the units"""
        return self._weight

    @property
    def terms(self) -> List:
        return self._terms

    @property
    def max_order(self) -> int:
        return self._max_order

    @property
    def variables(self) -> Set[Tuple]:
        """Get the indexes of all the unique moments used in this invariant e.g. if the invariant
        was:

            m_200 * m_200 * m_020 m_020 + m_002 m_002 + 2 m_110 m_110 + 2 m_101 m_101 + 2 m_011 m_011

        then this function would return

            {(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 0), (0, 1, 1)}

        as these are the all the moments (or variables) involved in calculating this invariant
        """
        variables = set()
        for _prefactor, product in self._terms:
            for part in product:
                variables.add(part)
        return variables

    def insert(self, prefactor, indices: Sequence[Tuple]):
        """
        :param prefactor: the prefactor for this term in the invariant
        :param indices: the indices of the moments involved in this invariant
        """
        if not indices:
            # If there are no indices supplied then it's just a constant
            self._constant += prefactor
            return

        if not all(len(entry) == 3 for entry in indices):
            raise ValueError('There have to be three indices per entry, got: {}'.format(indices))
        self._terms.append((prefactor, tuple(indices)))
        if indices:
            self._max_order = max(self._max_order, np.max(indices))

    def build(self):
        if self._terms:
            factors, arr = zip(*self._terms)
            self._farray = np.asarray(factors)
            self._indarray = np.asarray(arr)
            term = self._terms[0][1]
            self._norm_power = np.sum(term) / 3. + len(term)

    def apply(self, raw_moments: base_moments.Moments, normalise=False) -> float:
        """Compute this invariant from the given moments optionally normalising"""

        if isinstance(raw_moments, np.ndarray):
            total = self._numpy_apply(raw_moments)
        else:
            # If we can get a matrix we can still use the fast (numpy) method
            try:
                mtx = raw_moments.to_matrix()
            except AttributeError:
                # Ok, use generic method
                total = self._generic_apply(raw_moments)
            else:
                total = self._numpy_apply(mtx)

        if normalise:
            return total / raw_moments[0, 0, 0]**self._norm_power

        return total

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    # def _numpy_apply(self, raw_moments: np.ndarray):
    #     """Fast method to get the invariant from a numpy array"""
    #     total = self._constant  # type: float
    #
    #     if self._terms:
    #         indices = self._indarray
    #         total += np.prod(
    #             self._farray,
    #             np.product(raw_moments[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]],
    #                           axis=1)
    #         )
    #
    #     return total

    def _numpy_apply(self, raw_moments: np.ndarray):
        """Fast method to get the invariant from a numpy array"""
        total = self._constant  # type: float

        if self._terms:
            if len(self._terms) < 48:
                total += _numpy_apply(self._farray, self._indarray, raw_moments)
            else:
                total += _parallel_apply(self._farray, self._indarray, raw_moments)

        return total

    def _generic_apply(self, moments):
        """Generic apply for moments that support indexing.

        This is slower version of above but compatible with moments that aren't numpy arrays"""
        total = self._constant
        for factor, indices in self._terms:
            product = 1
            for index in indices:
                product *= moments[index]
            total += factor * product
        return total

    def derivatives(self) -> Dict[Tuple, 'MomentInvariant']:
        """Get analytical derivatives for this invariant wrt each of its variables

        Returns a dictionary whose key is the variable (the index tuple) and the value the
        corresponding MomentInvariant
        """
        if not self._derivatives:
            # Have to calculate first time
            deriv_terms = {}  # type: Dict[Tuple, MomentInvariant]
            for prefactor, product in self._terms:
                powers = collections.defaultdict(int)
                for indices in product:
                    powers[indices] += 1

                # Now carry out the analytical derivative wrt each of our variables
                for variable in self.variables:
                    if variable not in powers:
                        continue

                    derivative = deriv_terms.setdefault(variable, MomentInvariant(self._weight - 1))
                    new_product = []

                    # Multiply the prefactor by the current power of the variable
                    power = powers[variable]
                    # Calculate the new prefactor
                    new_prefactor = prefactor * power

                    # And add in the correct multiple of this variable
                    if power != 1:
                        new_product.extend((variable,) * (power - 1))

                    for var, power in powers.items():
                        if var == variable:
                            # Skip this, we've dealt with it above
                            continue

                        # The other terms keep their exponent unchanged
                        new_product.extend((var,) * power)

                    derivative.insert(new_prefactor, new_product)

            for entry in deriv_terms.values():
                entry.build()

            self._derivatives = deriv_terms

        return self._derivatives

    @staticmethod
    def _collect_powers(product: List[Tuple]) -> Dict[Tuple, int]:
        powers = collections.defaultdict(int)
        for indices in product:
            powers[indices] += 1
        return powers


def _numpy_apply(prefactors, indices: np.array, raw_moments: np.ndarray):
    """Fast method to get the invariant from a numpy array"""
    total = 0

    total += np.dot(prefactors, np.prod(raw_moments[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]], axis=1))

    return total


@numba.jit(parallel=True)
def _parallel_apply(prefactors, indices, moments):
    """Generic apply for moments that support indexing.

    This is slower version of above but compatible with moments that aren't numpy arrays"""
    total = 0.
    for idx in numba.prange(len(prefactors)):
        factor = prefactors[idx]
        entry = indices[idx]

        product = 1.
        for index in entry:
            product *= moments[index[0], index[1], index[2]]

        total += factor * product
    return total


class MomentInvariants(functions.Function):
    """A function that takes moments as input and produces rotation invariants using polyonimals thereof"""
    input_type = base_moments.Moments
    output_type = np.ndarray
    supports_jacobian = True

    def __init__(self, *invariant: MomentInvariant):
        super().__init__()
        for entry in invariant:
            if not isinstance(entry, MomentInvariant):
                raise TypeError(f'Expected MomentInvariant, got {entry.__class__.__name__}')

        self._invariants: List[MomentInvariant] = list(invariant)
        self._max_order = -1

    def __len__(self):
        return len(self._invariants)

    def __iter__(self) -> Iterator[MomentInvariant]:
        return self._invariants.__iter__()

    def __getitem__(self, item) -> Union['MomentInvariants', MomentInvariant]:
        if isinstance(item, slice):
            return MomentInvariants(*self._invariants[item])

        return self._invariants[item]

    @property
    def max_order(self) -> int:
        """Get the maximum order of all the invariants"""
        return self._max_order

    def output_length(self, in_state: functions.State) -> int:
        return len(self._invariants)

    def apply(self, moms: np.array, normalise=False, results=None) -> list:
        """Calculate the invariants from the given moments"""
        return apply_invariants(self._invariants, moms, normalise=normalise, results=results)

    def append(self, invariant: MomentInvariant):
        """Add an invariant"""
        self._invariants.append(invariant)
        self._max_order = max(self._max_order, invariant.max_order)

    def evaluate(self, moments: base_moments.Moments, get_jacobian=False) -> np.ndarray:
        vector = np.empty(len(self._invariants), dtype=np.promote_types(moments.vector.dtype, float))
        if get_jacobian:
            jac = np.zeros((len(self._invariants), len(moments)), dtype=vector.dtype)

        for idx, inv in enumerate(self._invariants):
            vector[idx] = inv.apply(moments, normalise=False)

            if get_jacobian:
                # Evaluate the derivatives
                for index, dphi in inv.derivatives().items():
                    in_index = moments.linear_index(index)
                    jac[idx, in_index] = dphi.apply(moments)

        if get_jacobian:
            return vector, jac

        return vector


def apply_invariants(invariants: List[MomentInvariant], moms: np.array, normalise=False, results=None) -> list:
    """Calculate the moment invariants for a given set of moments

    :param invariants: a list of invariants to calculate
    :param moms: the moments to use
    :param normalise: if True fill normalise the moments using the 0th moment
    :param results: an optional container to place the result in, if not supplied one will be created
    """
    if results is None:
        results = [None] * len(invariants)
    else:
        if not len(results) == len(invariants):
            raise ValueError('Results container must be of the same length as invariants')

    for idx, invariant in enumerate(invariants):
        results[idx] = invariant.apply(moms, normalise=normalise)

    return results


def read_invariants(filename: str = GEOMETRIC_INVARIANTS, read_max: int = None) -> \
        List[MomentInvariant]:
    """Read the Flusser, Suk and Zitová invariants.

    :param filename: the filename to read from, default to geometric moments invariants
    :param read_max: the maximum number of invariants to read
    """
    invariants = []
    with open(filename, 'r') as file:

        for line in file:
            line = line.rstrip()
            if line:
                # New entry
                header = [int(number) for number in line.split(' ')]
                num_terms = header[2]
                invariant = MomentInvariant(num_terms)

                # Now read the actual terms
                line = file.readline().rstrip()
                while line:
                    terms = tuple(map(str_to_number, line.split(' ')))

                    prefactor = terms[0]

                    indices = []
                    for idx in range(num_terms):
                        indices.append(tuple(terms[idx * 3 + 1:(idx + 1) * 3 + 1]))
                    invariant.insert(prefactor, indices)

                    line = file.readline().rstrip()

                invariant.build()
                invariants.append(invariant)
                if len(invariants) == read_max:
                    break

    return invariants


def str_to_number(value: str) -> Union[int, float, complex]:
    """Convert an integer, float or complex number string to the correct number type"""
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    try:
        return complex(value)
    except ValueError:
        pass

    raise ValueError('{} is not an int, float or complex'.format(value))


def read(filename: str = GEOMETRIC_INVARIANTS, read_max: int = None, max_order=None) -> \
        MomentInvariants:
    """Read the invariants from file"""
    invariants = MomentInvariants()
    if filename == COMPLEX_INVARIANTS:
        invariants.dtype = complex

    for inv in read_invariants(filename, read_max):
        if max_order is None or inv.max_order <= max_order:
            invariants.append(inv)
    return invariants


def calc_moment_invariants(
    invariants: Sequence[MomentInvariant],
    positions: np.array,
    sigma: Union[float, np.array] = 0.4,
    masses: Union[float, np.array] = 1.,
    normalise=False
) -> Sequence[float]:
    """Calculate the moment invariants for a set of Gaussians at the given positions."""
    max_order = 0

    # Calculate the maximum order invariant we'll need
    for inv in invariants:
        max_order = max(max_order, inv.max_order)

    raw_moments = geometric.from_gaussians(max_order, positions, sigma, masses)
    return tuple(invariant.apply(raw_moments, normalise) for invariant in invariants)
