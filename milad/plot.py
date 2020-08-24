# -*- coding: utf-8 -*-
import matplotlib.pyplot
import numpy


def plot_invariants(values,
                    axes: matplotlib.pyplot.Axes = None,
                    label=None,
                    scatter=True):
    if axes is None:
        _fig, axes = matplotlib.pyplot.subplots()

    kwargs = {}
    if label is not None:
        kwargs['label'] = label

    axes.plot(numpy.array(range(len(values))), values, **kwargs)

    if scatter:
        axes.scatter(numpy.array(range(len(values))), values, s=5.)

    return axes


def plot_multiple_invariants(series,
                             axes: matplotlib.pyplot.Axes = None,
                             labels='auto'):
    if axes is None:
        _fig, axes = matplotlib.pyplot.subplots()

    if labels == 'auto':
        labels = tuple('Series {}'.format(idx) for idx in range(len(series)))

    for idx, series in enumerate(series):
        kwargs = {}
        if labels is not None:
            kwargs['label'] = labels[idx]

        axes.plot(numpy.array(range(len(series))), series, **kwargs)

    axes.set_xlabel('Moment invariant #')
    axes.set_ylabel('Invariant value')
    return axes
