# -*- coding: utf-8 -*-
import sympy
from sympy.physics.quantum.cg import CG

from milad import utils


def composite_complex_moment_form(moments: sympy.Indexed, n: int, l: int, l_prime: int, j: int, k: int):
    """Get the composite moment form as defined by Lo and Don denoted as:

    c_n(l, l')_j^k
    """
    if not utils.even(j):
        raise ValueError('j must be even, got {}'.format(j))
    eqn = None

    for m in utils.inclusive(max(-l, k - l_prime), min(l, k + l_prime)):
        term = CG(l, m, l_prime, k - m, j, k).doit() * moments[n, l, m] * moments[n, l, -m]
        eqn = term if eqn is None else eqn + term

    assert eqn is not None
    return eqn


def moment_form_complex_moment(moments: sympy.Indexed, n: int, n_prime: int, l: int, l_prime: int, j: int):
    """Get a moment form combined with complex moments i.e.:

    c_s(l, l')_j c_{n'}
    """
    eqn = None

    for k in utils.inclusive(-j, j):
        term = (-1)**(j - k) * composite_complex_moment_form(moments, n, l, l_prime, j, k) * moments[n_prime, j, -k]
        eqn = term if eqn is None else eqn + term

    return 1 / sympy.sqrt(2 * j + 1) * eqn


def moment_form_moment_form(
    moments: sympy.Indexed, n: int, n_prime: int, l: int, l_prime: int, l_prime_prime: int, l_prime_prime_prime: int,
    j: int
):
    """Get a complex moment form combined with a complex moment form i.e.:

    c_n(l, l')_j c_{n'}(l'', l''')_j
    """
    eqn = None

    for k in utils.inclusive(-j, j):
        term = (-1) ** (j - k) * \
               composite_complex_moment_form(moments, n, l, l_prime, j, k) * \
               composite_complex_moment_form(moments, n_prime, l_prime_prime, l_prime_prime_prime, j, -k)
        eqn = term if eqn is None else eqn + term

    return 1 / sympy.sqrt(2 * j + 1) * eqn


def invariants_single_complex_moments(moments: sympy.Indexed, max_n: int):
    """Get invariants from single complex moments i.e.:

    c_{n0}^0
    """
    invariants = []
    for n in utils.inclusive(0, max_n, 2):
        invariants.append(moments[n, 0, 0])
    return invariants


def invariants_single_complex_moment_forms(moments: sympy.Indexed, max_n: int):
    """Get invariants from single complex moment forms i.e.:

    c_n(l, l)_0^0

    n = 1, 2, .., max_n
    l = n, n - 2, n - 4, ..., 1
    """
    invariants = []
    for n in utils.inclusive(1, max_n):
        for l in utils.inclusive(n, 1, -2):
            invariant = composite_complex_moment_form(moments, n=n, l=l, l_prime=l, j=0, k=0)
            invariants.append(invariant)
    return invariants


def invariants_moment_form_complex_moment(moments: sympy.Indexed, max_n: int):
    """Get invariants from a complex moment form and a complex moment i.e.:


    c_n(l, l')_j c_{n'}
    """
    invariants = []
    for n in utils.inclusive(1, max_n):
        for n_prime in utils.inclusive(2, n, 2):
            for j in utils.inclusive(2, n_prime, 2):
                for l in utils.inclusive(n, 1, -2):
                    for l_prime in utils.inclusive(l, max(l - j, 1), -2):
                        assert l_prime >= l - j
                        assert l_prime >= 1
                        invariant = moment_form_complex_moment(moments, n=n, n_prime=n_prime, l=l, l_prime=l_prime, j=j)
                        invariants.append(invariant)

    return invariants


def invariants_two_moment_forms(moments: sympy.Indexed, max_n: int):
    """Get invariants from two complex moment forms i.e.:


    c_n(l, l')_j c_{n'}(l'', l''')_j
    """
    invariants = []
    for n in utils.inclusive(1, max_n):
        for n_prime in utils.inclusive(1, n):
            for j in utils.inclusive(2, n_prime, 2):
                for l in utils.inclusive(n, 1, -2):
                    for l_prime in utils.inclusive(l, max(l - j, 1), -2):
                        for l_prime_prime in utils.inclusive(n_prime, 1, -2):
                            for l_prime_prime_prime in utils.inclusive(l_prime_prime, max(l_prime_prime - j, 1), -2):
                                invariant = moment_form_moment_form(
                                    moments,
                                    n=n,
                                    n_prime=n_prime,
                                    l=l,
                                    l_prime=l_prime,
                                    l_prime_prime=l_prime_prime,
                                    l_prime_prime_prime=l_prime_prime_prime,
                                    j=j
                                )
                                invariants.append(invariant)
    return invariants
