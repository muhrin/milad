import numpy

__all__ = 'norm_moment', 'multidim_norm_moments', 'moment_tensor3d', 'calc_raw_moments3d'


def norm_moment(mu: float, sigma: float, order: int) -> float:
    """Get the nt^h moment of a 1D normal distribution centred at `mu`
    with a standard deviation of `sigma`

    Taken from https://en.wikipedia.org/wiki/Normal_distribution#Moments
    Can be generalised to any order using confluent hypergeometric functions of the second kind

    :param mu: the mean of the distribution
    :param sigma: the standard deviation of the distribution
    :param order: the order of the moment to get
    """
    if order == 0:
        return 1.
    if order == 1:
        return mu
    if order == 2:
        return mu ** 2 + sigma ** 2
    if order == 3:
        return mu ** 3 + 3 * mu * sigma ** 2
    if order == 4:
        return mu ** 4 + 6 * mu ** 2 * sigma ** 2 + 3 * sigma ** 4
    if order == 5:
        return mu ** 5 + 10 * mu ** 3 * sigma ** 2 + 15 * mu * sigma ** 4
    if order == 6:
        return mu ** 6 + 15 * mu ** 4 * sigma ** 2 + 45 * mu ** 2 + sigma ** 4 + 15 * sigma ** 6
    if order == 7:
        return mu ** 7 + 21 * mu ** 5 * sigma ** 2 + 105 * mu ** 3 * sigma ** 4 + 105 * mu * sigma ** 6
    if order == 8:
        return mu ** 8 + 28 * mu ** 6 * sigma ** 2 + 210 * mu ** 4 * sigma ** 4 + 420 * mu ** 2 * sigma ** 6 + 105 * sigma ** 8

    raise NotImplemented("Asked for order '{}', only up to order 8 implemented!".format(order))


def multidim_norm_moments(mu: numpy.array, sigma: float, order: int) -> numpy.array:
    """Calculate the moments for a multidimensional normal distribution.  The distribution is
    symmetric with standard deviation `sigma` centred at `mu`.  The `order`^th order will be
    returned.

    :param mu: the centre of the distribution
    :param sigma: the standard deviation of the distribution
    :param order: the order of the moments to return
    """
    return numpy.array([norm_moment(entry, sigma, order) for entry in mu])


def moment_tensor3d(
        mu: numpy.array,
        sigma: float,
        max_order: int,
        scale: float = 1.0,
        normalise=True) -> numpy.array:
    """
    :param mu: the position of the normal distribution
    :param sigma: the standard deviation (scalar - same in all directions)
    :param max_order: the maximum order to calculate moments for
    :param scale: the area of the gaussian
    :param normalise: divide the moments by the scale to the power of the order effectively making
        them unitless
    :return: the 3d moment tensor
    """
    moments = numpy.zeros((3, max_order))
    moments[:, 0] = scale
    for order in range(1, max_order):
        normaliser = 1.0 if not normalise else sigma ** order
        moments[:, order] = multidim_norm_moments(mu, sigma, order) / normaliser

    mom_tensor = numpy.empty((max_order,) * 3)
    for i in range(max_order):
        for j in range(max_order):
            for k in range(max_order):
                mom_tensor[i, j, k] = moments[0, i] * moments[1, j] * moments[2, k]

    return mom_tensor


def calc_raw_moments3d(
        positions: numpy.array,
        sigma: float,
        max_order: int,
        scale: float = 1.,
        normalise=True):
    """Calculate the raw moments tensor for a collection of Gaussians at the given positions with
    the passed parameters.

    :param positions: the positions of the Gaussians
    :param sigma: the standard deviation of the gaussians
    :param max_order: the maximum order to calculate moments up to
    :param scale: the area of the Gaussians
    :param normalise: if True, normalise the moments by dividing the moments by the scale raise to
        the power of the order.  This renders them unitless.
    """
    moments = numpy.zeros((max_order, max_order, max_order))
    for pos in positions:
        moments += moment_tensor3d(pos, sigma, max_order, scale, normalise)
    return moments
