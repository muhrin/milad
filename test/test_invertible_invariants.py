# -*- coding: utf-8 -*-
import numpy as np

from milad import invertible_invariants
from milad import zernike
import milad


def nearest_postdef(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    # pylint: disable=invalid-name

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_posdef(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_posdef(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_posdef(B):
    """Returns true when input is positive-definite, via Cholesky"""
    # pylint: disable=invalid-name

    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


class test_invertible_invariants_basics():
    n_max = 7

    generator = invertible_invariants.InvariantsGenerator()

    # Generate the invariants
    invs = generator.generate_all(n_max)

    # Create some random moments and calculate the fingerprint
    rand_moms = zernike.from_deltas(n_max, milad.generate.random_points_in_sphere(5))
    # rand_moms = zernike.rand(n_max)
    phi = invs(rand_moms)

    x1 = rand_moms.array[1:, 1, (-1, 0, 1)][(0, 2, 4, 6), :].data
    G = x1 @ x1.conj().T
    # L = np.linalg.cholesky(G)

    vals = np.linalg.eigvals(G)
    print(vals)

    inverted = zernike.ZernikeMoments(n_max)
    invs.invert(phi, inverted)


def test_play():
    for _ in range(10):
        n_max = 7

        rand_moms = zernike.from_deltas(n_max, milad.generate.random_points_in_sphere(5))

        x1 = rand_moms.array[1:, 1, (-1, 0, 1)][(0, 2, 4, 6), :].data.T
        G = x1.conj().T @ x1
        # L = np.linalg.cholesky(G)

        G = nearest_postdef(G)

        vals = np.linalg.eigvals(G)
        print(vals[vals < 0])
