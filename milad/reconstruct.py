# -*- coding: utf-8 -*-
import functools
import collections
import logging

import numpy as np
import rmsd as rmsdlib
from sklearn import cluster

from . import atomic
from . import base_moments
from . import exceptions
from . import fingerprinting
from . import optimisers

__all__ = ('Decoder',)

_LOGGER = logging.getLogger(__name__)

StructureOptimisationResult = collections.namedtuple('StructureOptimisationResult', 'success atoms message rmsd')


@functools.singledispatch
def find_clusters(spec, num_clusters: int, **kwargs) -> np.ndarray:
    """Find clusters based on moments"""
    raise TypeError(f'Cannot find clusters from {spec.__class__.__name__}')


@find_clusters.register(base_moments.Moments)
def _(moments: base_moments.Moments, num_clusters: int, query: base_moments.ReconstructionQuery,
      fingerprinter: fingerprinting.MomentInvariantsDescriptor) -> np.ndarray:
    """Find clusters from moments.  This will take the moments and reconstruct values on a grid
    which will be used for the actual cluster determination"""
    # Calculate the grid values
    values = moments.reconstruct(query, zero_outside_domain=True)
    # Find the cluster centres
    return find_clusters((query.points, values), num_clusters)


@find_clusters.register(tuple)
def _(grid, num_clusters: int) -> np.ndarray:
    """Given a grid consisting of points and weights this function funds clusters using k-means"""
    grid_points, grid_values = grid
    kmeans = cluster.KMeans(num_clusters, max_iter=1000, algorithm='full', tol=1e-1)
    kmeans.fit(grid_points, sample_weight=grid_values)

    num_labels = len(set(kmeans.labels_))
    if num_labels != num_clusters:
        raise exceptions.ReconstructionError(f'Could not find {num_clusters} clusters, found {num_labels} labels')
    if len(kmeans.cluster_centers_) != num_clusters:
        raise exceptions.ReconstructionError(
            f'Could not find {num_clusters} clusters, found {len(kmeans.cluster_centers_)} centres'
        )

    return kmeans.cluster_centers_


def find_peaks(
    moments: base_moments.Moments, num_peaks: int, query: base_moments.ReconstructionQuery,
    descriptor: fingerprinting.MomentInvariantsDescriptor
):
    atom_positions = []

    current_grid = moments.reconstruct(query, zero_outside_domain=True)
    for _ in range(num_peaks):
        # Fing the index of the maximum value in the current grid
        max_idx = current_grid.argmax()

        # Get that position in the grid
        atom_pos = query.points[max_idx]
        # Build an atoms collection with a single atom at that position
        single_atom = atomic.AtomsCollection(1, positions=[atom_pos], numbers=[1.])
        if descriptor.scaler is not None:
            single_atom = descriptor.scaler.inverse(single_atom)

        atom_positions.append(single_atom.positions[0])

        # Get the moments so we can subtract this from the grid
        single_moments = descriptor.get_moments(single_atom, preprocess=False)

        # Get the grid for just that atom on its own
        atom_grid = single_moments.reconstruct(query, zero_outside_domain=True)

        # Subract off the single atom grid
        current_grid -= atom_grid

        # Now remove the signal of the atom from the grid
        remove_idxs = np.argwhere(atom_grid >= (0.5 * atom_grid.max()))
        current_grid[remove_idxs] = 0.

    return np.array(atom_positions)


def create_atoms_collection(clusters: cluster.KMeans, atomic_numbers=1.):
    """Take a set of clusters and use the centres to construct an atoms collection"""
    num_atoms = len(clusters.cluster_centers_)
    return atomic.AtomsCollection(num_atoms, positions=clusters.cluster_centers_, numbers=atomic_numbers)


class Decoder:
    """This class decodes a structure from a set of moment invariants"""

    def __init__(
        self,
        descriptor: fingerprinting.MomentInvariantsDescriptor,
        moments_query=None,
        initial_finder=find_peaks,
        default_grid_size=31
    ):
        self._descriptor = descriptor
        self._optimiser = optimisers.StructureOptimiser()
        self._moments_query = moments_query
        self._initial_finder = initial_finder
        self._default_grid_size = default_grid_size

    def decode(
        self,
        invariants: np.ndarray,
        moments: base_moments.Moments,
        num_atoms: int,
        atomic_numbers=1.
    ) -> StructureOptimisationResult:
        if self._moments_query is None:
            query = moments.create_reconstruction_query(moments.get_grid(self._default_grid_size), moments.max_order)
        else:
            query = self._moments_query

        # Get the clusters from the moments, the positions will be in the range [-1, 1]
        positions = self._initial_finder(moments, num_atoms, query, self._descriptor)
        if positions.min() < -1 or positions.max() > 1:
            raise exceptions.ReconstructionError('Clustering algorithm returned centres that are out of bounds')

        initial_guess = atomic.AtomsCollection(num_atoms, positions=positions, numbers=atomic_numbers or 1.)
        # Remap the starting configuration back to the correct size
        if self._descriptor.scaler is not None:
            initial_guess = self._descriptor.scaler.inverse(initial_guess)

        return self._optimiser.optimise(descriptor=self._descriptor, target=invariants, initial=initial_guess)


def get_best_rms(
    reference: atomic.AtomsCollection,
    probe: atomic.AtomsCollection,
    max_attempts: int = 1000,
    max_retries=20,
    threshold=1e-7,
) -> float:
    """
    Get the best RMSs fitting between two molecules.  This will first use an algorithm to make a decent guess at the
    best permutational ordering of atoms and then try a brute force search.

    :param reference: the reference to fit to
    :param probe: the probe to try to fit to the reference
    :return: the best RMSD found
    """
    from rdkit.Chem import rdMolAlign
    from . import rdkittools

    # Find a decent re-ordering of the atoms
    try:
        reorder_map = rmsdlib.reorder_hungarian(reference.numbers, probe.numbers, reference.positions, probe.positions)
    except ValueError:
        pass
    else:
        probe = atomic.AtomsCollection(
            reference.num_atoms,
            positions=probe.positions[reorder_map],
            numbers=probe.numbers[reorder_map],
        )

    # Now ask rdkit to find the best.
    # This will try a brute-force permutation search starting with the one we just defined which is likely to be
    # the best
    reference = rdkittools.milad2rdkit(reference)
    probe = rdkittools.milad2rdkit(probe)
    try:
        best = np.inf
        for _ in range(max_retries):
            best = min(rdMolAlign.GetBestRMS(probe, reference, maxMatches=max_attempts), best)
            if best < threshold:
                break
        return best
    except RuntimeError:
        return np.inf
