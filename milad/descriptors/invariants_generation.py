# -*- coding: utf-8 -*-
import sympy
from sympy.physics.quantum.cg import CG

from milad import utils

# We're using a lot of variable names that make sense from the mathematics but are not so programming friendly
# so disable for now as we know what they mean
# pylint: disable=invalid-name


def composite_complex_moment_form(moments: sympy.Indexed, n: int, l1: int, l2: int, j: int, k: int):
    """Get the composite moment form as defined by Lo and Don denoted as:

    c_n(l, l')_j^k
    """
    if not l1 >= 0:
        raise ValueError(f'l must be greater than 0, got {l1}')
    if not l2 >= 0:
        raise ValueError(f'l2 must be greater than 0, got {l2}')
    if not utils.even(j):
        raise ValueError('j must be even, got {}'.format(j))
    if not utils.even(n - l1):
        raise ValueError(f'n - l must be even got {n} - {l1}')
    if not utils.even(n - l2):
        raise ValueError(f'n - l2 must be even got {n} - {l2}')
    eqn = None

    for m in utils.inclusive(max(-l1, k - l2), min(l1, k + l2), 1):
        term = CG(l1, m, l2, k - m, j, k).doit() * moments[n, l1, m] * moments[n, l2, k - m]
        eqn = term if eqn is None else eqn + term

    assert eqn is not None, (max(-l1, k - l2), min(l1, k + l2))
    return eqn


def moment_form_complex_moment(moments: sympy.Indexed, n1: int, l1: int, l2: int, n2: int, j: int):
    """Get a moment form combined with complex moments i.e.:

    c_s(l, l')_j c_{n'}
    """
    if not j >= 0:
        raise ValueError(f"'j' must be >= 0, got {j}")
    if not utils.even(n2 - j):
        raise ValueError(f'n2 - j must be even got {n2} - {j}')

    eqn = None

    for k in utils.inclusive(-j, j, 1):
        term = (-1)**(j - k) * composite_complex_moment_form(moments, n1, l1, l2, j, k) * moments[n2, j, -k]
        eqn = term if eqn is None else eqn + term

    return 1 / sympy.sqrt(2 * j + 1) * eqn


def moment_form_moment_form(moments: sympy.Indexed, n1: int, l1: int, l2: int, n2: int, l3: int, l4: int, j: int):
    """Get a complex moment form combined with a complex moment form i.e.:

    c_n(l, l')_j c_{n'}(l'', l''')_j
    """
    eqn = None

    for k in utils.inclusive(-j, j, 1):
        term = (-1) ** (j - k) * \
               composite_complex_moment_form(moments, n1, l1, l2, j, k) * \
               composite_complex_moment_form(moments, n2, l3, l4, j, -k)
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
    for n in utils.inclusive(1, max_n, 1):
        for l in utils.inclusive(n, 1, -2):
            invariant = composite_complex_moment_form(moments, n=n, l1=l, l2=l, j=0, k=0)
            invariants.append(invariant)
    return invariants


def invariants_moment_form_complex_moment(moments: sympy.Indexed, max_n: int):
    """Get invariants from a complex moment form and a complex moment i.e.:


    c_n(l, l')_j c_{n'}
    """
    invariants = []
    for n in utils.inclusive(1, max_n, 1):
        for n2 in utils.inclusive(2, n, 2):
            for j in utils.inclusive(2, n2, 2):
                for l1 in utils.inclusive(n, 1, -2):
                    for l2 in utils.inclusive(l1, max(l1 - j, 1), -2):
                        assert l2 >= l1 - j
                        assert l2 >= 1
                        invariant = moment_form_complex_moment(moments, n1=n, l1=l1, l2=l2, n2=n2, j=j)
                        invariants.append(invariant)

    return invariants


def invariants_two_moment_forms(moments: sympy.Indexed, max_n: int, verbose=False):
    """Get invariants from two complex moment forms i.e.:


    c_n(l, l')_j c_{n'}(l'', l''')_j
    """
    invariants = []
    for n in utils.inclusive(1, max_n, 1):
        for n_prime in utils.inclusive(1, n, 1):
            for j in utils.inclusive(0, n_prime, 2):
                for l in utils.inclusive(n, 1, -2):
                    for l2 in utils.inclusive(l, max(l - j, 1), -2):
                        for l3 in utils.inclusive(n_prime, 1, -2):
                            for l4 in utils.inclusive(l3, max(l3 - j, 1), -2):
                                invariant = moment_form_moment_form(
                                    moments, n1=n, n2=n_prime, l1=l, l2=l2, l3=l3, l4=l4, j=j
                                )
                                if verbose:
                                    print(f'{len(invariants)} {n} {l} {l2} ')
                                invariants.append(invariant)
    return invariants


def get_invariant_header(invariant):
    coeffs_dict = invariant.expand().evalf().as_coefficients_dict()
    parts = tuple(coeffs_dict)

    if isinstance(parts[0], sympy.Mul):
        terms = parts[0].args
    else:
        terms = [parts[0]]

    power = 0
    weight = 0
    for term in terms:
        print(term)
        if isinstance(term, sympy.Pow):
            exponent = term.args[1]
            power += exponent
            weight += exponent * term.args[0].indices[0]
        else:
            power += 1
            weight += term.indices[0]

    return f'{str(weight / 2)} {len(coeffs_dict)} {power}'


def get_string_block(invariant):
    coeffs_dict = invariant.expand().evalf().as_coefficients_dict()
    lines = [get_invariant_header(invariant)]

    for terms, coeff in coeffs_dict.items():
        parts = []
        parts.append(str(coeff))

        if isinstance(terms, sympy.Mul):
            pts = terms.args
        else:
            pts = [terms]

        for term in pts:
            if isinstance(term, sympy.Pow):
                symbol, power = term.args
                indices = symbol.indices
                parts.extend([f'{indices[0]} {indices[1]} {indices[2]}'] * power)
            else:
                indices = term.indices
                parts.append(f'{indices[0]} {indices[1]} {indices[2]}')
        lines.append(' '.join(parts))

    return '\n'.join(lines)
