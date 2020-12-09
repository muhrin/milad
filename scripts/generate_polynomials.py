# -*- coding: utf-8 -*-
from typing import Sequence

import numpy as np

import milad
from milad.play import SmoothGaussianEnvironment

np.random.seed(123)


def indices_to_str(indices):
    # First collect all the powers together
    powers = {}
    for index in indices:
        joined = ','.join(map(str, index))
        powers.setdefault(joined, 0)
        powers[joined] += 1

    # Now create the factor string
    strs = []
    for joined, power in powers.items():
        term = ['m', ''.join(joined.split(','))]
        if power > 1:
            term.append('^{}'.format(power))
        strs.append(''.join(term))

    return '*'.join(strs)


def terms_eq_value(terms, value):
    return '{} = {}\n'.format(' + '.join(terms), value)


def generate_polys(moments, invs: Sequence[milad.invariants.MomentInvariant], d_max: int):
    polys = []
    for inv in invs:
        if inv.max_order > d_max:
            continue

        terms = []
        for coeff, indices in inv.terms:
            term = []
            if coeff > 1:
                term.append(str(coeff))
            term.append(indices_to_str(indices))
            terms.append('*'.join(term))

        phi = inv.apply(moments, normalise=False)
        polys.append(terms_eq_value(terms, phi))

    return polys


if __name__ == '__main__':
    num_atoms = 5
    scale = 4.
    sigma = 0.5
    cutoff = 6.
    d_max = 4

    invs = milad.invariants.read_invariants()

    positions = scale * np.random.rand(num_atoms, 3)

    env = SmoothGaussianEnvironment(cutoff=cutoff)
    for pos in positions:
        env.add_gaussian(pos, sigma)

    moments = env.moment_tensor(d_max)

    polys = []
    # Create the first terms
    polys.extend([
        terms_eq_value([indices_to_str([[2, 0, 0]])], moments[2, 0, 0]),
        terms_eq_value([indices_to_str([[0, 2, 0]])], moments[0, 2, 0]),
        terms_eq_value([indices_to_str([[0, 0, 2]])], moments[0, 0, 2]),
    ])

    polys.extend(generate_polys(moments, invs, d_max))

    with open('polys.txt', 'w') as polys_out:
        polys_out.writelines(polys)
