# -*- coding: utf-8 -*-
import math
import functools

import numpy
import scipy
import scipy.special

# pylint: disable=invalid-name

__all__ = 'calc_zernike_moments', 'ZernikeMoments'


@functools.lru_cache(maxsize=None)
def factorial(n: int) -> int:
    return scipy.special.factorial(n, exact=True)


@functools.lru_cache(maxsize=None)
def binomial_coeff(n: int, k: int) -> int:
    return scipy.special.comb(n, k, exact=True)


@functools.lru_cache(maxsize=None)
def c_l_m(l: int, m: int) -> float:
    """Calculate the normalisation factor"""
    if m < 0:
        # make use of symmetry c_l^m == c_l^-m
        return c_l_m(l, -m)

    return (factorial(2 * l + 1) * factorial(l + m) *
            factorial(l - m))**0.5 / factorial(l)


@functools.lru_cache(maxsize=None)
def q_kl_nu(k: int, l: int, nu: int) -> float:
    """Calculate the Zernike geometric moment conversion factors"""
    return (-1) ** k / 2 ** (2 * k) * \
           ((2 * l + 4 * k + 3) / 3) ** 0.5 * \
           binomial_coeff(2 * k, k) * \
           (-1) ** nu * \
           (binomial_coeff(k, nu) * binomial_coeff(2 * (k + l + nu) + 1, 2 * k)) / \
           binomial_coeff(k + l + nu, k)


def chi_nlm(n: int, l: int, m: int, geom_moments: numpy.ndarray) -> complex:
    """Calculate the Zernike geometric moment conversion factors"""
    return c_l_m(l, m) * 2**(-m) * sum1(
        n=n, l=l, m=m, geom_moments=geom_moments)


def sum1(n: int, l: int, m: int, geom_moments: numpy.ndarray) -> complex:
    """Calculate the Zernike geometric moment conversion factors"""
    k = int((n - l) / 2)  # n - l is always even
    total = 0.
    for nu in range(0, k + 1):
        total += q_kl_nu(k, l, nu) * sum2(
            n=n, l=l, m=m, nu=nu, geom_moments=geom_moments)

    return total


def sum2(n: int, l: int, m: int, nu: int,
         geom_moments: numpy.ndarray) -> complex:
    total = 0.
    for alpha in range(0, nu + 1):
        total += binomial_coeff(nu, alpha) * \
                 sum3(n=n, l=l, m=m, nu=nu, alpha=alpha, geom_moments=geom_moments)
    return total


def sum3(n: int, l: int, m: int, nu: int, alpha: int,
         geom_moments: numpy.ndarray) -> complex:
    total = 0.
    for beta in range(0, nu - alpha + 1):
        total += binomial_coeff(nu - alpha, beta) * \
                 sum4(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, geom_moments=geom_moments)
    return total


def sum4(n: int, l: int, m: int, nu: int, alpha: int, beta: int,
         geom_moments: numpy.ndarray) -> complex:
    total = 0.
    for u in range(0, m + 1):
        total += (-1) ** (m - u) * binomial_coeff(m, u) * 1j ** u * \
                 sum5(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, u=u,
                      geom_moments=geom_moments)
    return total


def sum5(n: int, l: int, m: int, nu: int, alpha: int, beta: int, u: int,
         geom_moments: numpy.ndarray) -> complex:
    total = 0.
    for mu in range(0, int((l - m) / 2.) + 1):
        total += (-1) ** mu * 2 ** (-2 * mu) * binomial_coeff(l, mu) * \
                 binomial_coeff(l - mu, m + mu) * \
                 sum6(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, u=u, mu=mu,
                      geom_moments=geom_moments)
    return total


def sum6(n: int, l: int, m: int, nu: int, alpha: int, beta: int, u: int,
         mu: int, geom_moments: numpy.ndarray) -> complex:
    total = 0.
    for v in range(0, mu + 1):
        r = 2 * (v + alpha) + u
        s = 2 * (mu - v + beta) + m - u
        t = 2 * (nu - alpha - beta - mu) + l - m

        if r + s + t > n:
            continue

        total += binomial_coeff(mu, v) * geom_moments[r, s, t]
    return total


def assert_valid(n: int, l: int, m: int):
    if not n >= 0:
        raise ValueError('n must be a positive integer')
    if not 0 <= l <= n:
        raise ValueError('l must be 0 <= l <= n')
    if not -l <= m <= l:
        raise ValueError('m must be in the range -l <= m <= l')
    if not ((n - l) % 2) == 0:
        raise ValueError('n - l must be even')


def omega_nl_m(n: int, l: int, m: int, geom_moments: numpy.array) -> complex:
    """Given a set of geometric moments this function will compute the corresponding Zernike moments
    """
    assert_valid(n, l, m)
    return 3 / (4 * math.pi) * chi_nlm(n, l, m, geom_moments)


class ZernikeMoments:
    """A container class for calculated Zernike moments"""
    def __init__(self, n_max: int, geom_coeffs: numpy.array):
        self._max_n = n_max
        self._geom_coeffs = geom_coeffs
        self._moments = numpy.empty((n_max + 1, n_max + 1, n_max + 1),
                                    dtype=complex)

        # Fill with a number I will recognise if it's still left there
        # (which it shouldn't be for valid indexes)
        self._moments.fill(1985)

        # Let's calculate all the moments (for positive m)
        for n in range(0, n_max + 1):
            for l in range(0, n + 1):
                if ((n - l) % 2) != 0:
                    continue
                for m in range(0, l + 1):
                    self._moments[n, l, m] = omega_nl_m(n, l, m, geom_coeffs)

    def moment(self, n: int, l: int, m: int) -> complex:
        """Get the n, l, m^th moment"""
        assert_valid(n, l, m)
        m_prime = abs(m)
        value = self._moments[n, l, m_prime]
        if m < 0:
            value = (-1)**m_prime * value.conjugate()
        return value


def calc_zernike_moments(n_max, geom_coeffs: numpy.array) -> ZernikeMoments:
    return ZernikeMoments(n_max, geom_coeffs)
