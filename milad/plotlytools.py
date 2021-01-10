# -*- coding: utf-8 -*-
import numpy as np
import plotly.graph_objects as go


def isosurface(grid: np.ndarray, values: np.ndarray, **kwargs) -> go.Isosurface:
    """Helper for creating a plotly isosurface from an array of grid coordinates and corresponding values"""
    params = dict(isomax=np.inf)
    params.update(kwargs)

    return go.Isosurface(x=grid[:, 0], y=grid[:, 1], z=grid[:, 2], value=values, **params)


def scatter3d(points: np.ndarray, **kwargs) -> go.Scatter3d:
    """Helper for creating a plotly 3d scatter plot from an array of points"""
    params = dict(mode='markers')
    params.update(kwargs)
    return go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], **params)
