# -*- coding: utf-8 -*-
import numpy as np


def to_real(vec: np.array) -> np.array:
    """Convert a given numpy vector containing complex numbers to one twice as long containing
    only real numbers where the first half contains the real and the second half the imaginary parts
    """
    view = vec.view('(2,)float')
    real = view.reshape(view.shape[0] * view.shape[1])
    return real


def to_complex(vec: np.array) -> np.array:
    """Given a vector of real numbers convert it to one half the size containing complex numbers
    where the first half of the original vector is treated as the real parts while the second half
    is used for the imaginary"""
    half_size = int(vec.size / 2)
    reshaped = vec.reshape((half_size, 2))
    view = reshaped.view(dtype=complex).reshape(half_size)

    return view


def even(val: int) -> bool:
    """Test if an integer is event.  Returns True if so."""
    return (val % 2) == 0


def odd(val: int) -> bool:
    """Test if an integer is odd.  Returns True if so."""
    return (val % 2) != 0


def cholesky(gram: np.ndarray) -> np.array:
    """Find Cholesky decomposition of the passed Gram matrix.  If this fails the algorithm
    will attempt to force it to be positive definite

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988):
    https://doi.org/10.1016/0024-3795(88)90223-6
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
