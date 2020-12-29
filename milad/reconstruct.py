# -*- coding: utf-8 -*-
import functools
import collections
import logging
from typing import Tuple, Optional

import math
import numpy as np
import rmsd as rmsdlib
from scipy import optimize
from sklearn import cluster

from . import atomic
from . import base_moments
from . import exceptions
from . import fingerprinting
from . import functions

__all__ = ('StructureOptimiser',)

_LOGGER = logging.getLogger(__name__)

StructureOptimisationResult = collections.namedtuple('StructureOptimisationResult', 'success atoms message rmsd')


class StructureOptimiser:
    """
    The structure optimiser takes a fingerprinting function and performs a least squares optimistaion
    to match a structure to a given fingerprint.
    """

    def __init__(self, fingerprinter: fingerprinting.Fingerprinter):
        super().__init__()
        self._fingerprinter = fingerprinter

    @property
    def fingerprinter(self) -> fingerprinting.Fingerprinter:
        return self._fingerprinter

    def optimise(
        self,
        fingerprint: functions.StateLike,
        starting_configuration: atomic.AtomsCollection,
        xtol=1e-5,
        max_evaluations=5000,
        atoms_builder: atomic.AtomsCollectionBuilder = None,
    ) -> StructureOptimisationResult:
        """
        :param fingerprint: the fingerprint to decode back into an atoms collection
        :param starting_configuration: the starting atoms configuration
        :param xtol: stopping criterion for the fitting algorithm
        :param max_evaluations: the maximum number of allowed fingerprint evaluations
        :param atoms_builder: an optional atoms builder that can be used to freeze certain degrees of freedom
        :return: a structure optimisation result
        """
        preprocess = self._fingerprinter.preprocess

        if atoms_builder:
            preprocessed = preprocess(starting_configuration)
            fixed_indices = np.argwhere(atoms_builder.mask != None)  # pylint: disable=singleton-comparison
            builder = atomic.AtomsCollectionBuilder(starting_configuration.num_atoms)
            builder.mask[fixed_indices] = preprocessed.array[fixed_indices]
            atoms_builder = builder
        else:
            atoms_builder = atomic.AtomsCollectionBuilder(starting_configuration.num_atoms)

        # We're going to need a residuals function
        residuals = functions.Chain(atoms_builder, self._fingerprinter.process, functions.Residuals(fingerprint))

        previous_result: Optional[Tuple] = None

        def calc(state: functions.StateLike):
            global previous_result
            # Calculate residuals and Jacobian
            res, jac = residuals(state, jacobian=True)
            print(f'Decoding max(|R|)): {np.abs(res).max()}')

            _LOGGER.info('Decoding max(|R|)): %d', np.abs(res).max())
            previous_result = state.real, jac.real
            return res.real

        def jac(state: functions.StateLike):
            global previous_result
            if np.all(previous_result[0] == state):
                return previous_result[1]

            _, jacobian = residuals(state, jacobian=True)
            return jacobian.real

        # Preprocess the starting structure and get the corresponding flattened array
        preprocess = self._fingerprinter.preprocess
        preprocessed = preprocess(starting_configuration)
        starting_vec = functions.get_bare_vector(atoms_builder.inverse(preprocessed))

        result = optimize.least_squares(
            calc,
            x0=starting_vec,
            jac=jac,
            bounds=self._get_bounds(atoms_builder),
            xtol=xtol,
            max_nfev=max_evaluations,
        )

        # Build the atoms from the vector and then 'un-preprocess' it (likely scale size and map species)
        final_atoms = preprocess.inverse(atoms_builder(result.x))

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        # scipy uses the cost function:
        # F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
        # and so the RMSD is sqrt(2 / len(x) * F(x))
        rmsd = math.sqrt(2 / len(starting_vec) * result.cost)
        return StructureOptimisationResult(success=result.success, atoms=final_atoms, message=result.message, rmsd=rmsd)

    def _get_bounds(self, builder: atomic.AtomsCollectionBuilder) -> Tuple[np.ndarray, np.ndarray]:
        preprocess = self._fingerprinter.preprocess

        species_range = (-np.inf, np.inf)
        positions_range = (-np.inf, np.inf)
        results = preprocess.find_type(atomic.MapNumbers)
        if results:
            species_range = results[0][1].mapped_range
        if self._fingerprinter.cutoff is not None:
            # positions_range = (-self._fingerprinter.cutoff, self._fingerprinter.cutoff)
            positions_range = (-1, 1)

        # Create a dummy atoms we can use  for figuring out the range of the vector
        num_atoms = builder.num_atoms
        atoms = atomic.AtomsCollection(num_atoms)

        # Lower bounds
        atoms.positions = positions_range[0]
        atoms.numbers = species_range[0]
        lower = builder.inverse(atoms)

        # Upper
        atoms.positions = positions_range[1]
        atoms.numbers = species_range[1]
        upper = builder.inverse(atoms)

        return lower, upper


@functools.singledispatch
def find_clusters(spec, num_clusters: int, **kwargs) -> np.ndarray:
    """Find clusters based on moments"""
    raise TypeError(f'Cannot find clusters from {spec.__class__.__name__}')


@find_clusters.register(base_moments.Moments)
def _(moments: base_moments.Moments, num_clusters: int, query: base_moments.ReconstructionQuery,
      fingerprinter: fingerprinting.Fingerprinter) -> np.ndarray:
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
    fingerprinter: fingerprinting.Fingerprinter
):
    atom_positions = []

    current_grid = moments.reconstruct(query, zero_outside_domain=True)
    for _ in range(num_peaks):
        # Fing the index of the maximum value in the current grid
        max_idx = current_grid.argmax()

        # Get that position in the grid
        atom_pos = query.points[max_idx]
        atom_positions.append(atom_pos)

        # Build an atoms collection with a single atom at that position
        single_atom = atomic.AtomsCollection(1, positions=[atom_pos], numbers=[1.])
        single_moments = fingerprinter.get_moments(single_atom, preprocess=False)

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

    def __init__(self, fingerprinter: fingerprinting.Fingerprinter, moments_query=None, initial_finder=find_peaks):
        self._fingerprinter = fingerprinter
        self._optimiser = StructureOptimiser(fingerprinter)
        self._moments_query = moments_query
        self._initial_finder = initial_finder

    def decode(
        self, phi, moments: base_moments.Moments, num_atoms: int, atomic_numbers=None
    ) -> StructureOptimisationResult:
        if self._moments_query is None:
            query = moments.create_reconstruction_query(moments.get_grid(31), moments.max_order)
        else:
            query = self._moments_query

        # Get the clusters from the moments, the positions will be in the range [-1, 1]
        centres = self._initial_finder(moments, num_atoms, query, self._fingerprinter)

        if centres.min() < -1 or centres.max() > 1:
            raise exceptions.ReconstructionError('Clustering algorithm returned centres that are out of bounds')

        from_centres = atomic.AtomsCollection(num_atoms, positions=centres, numbers=atomic_numbers or 1.)

        # Remap the starting configuration back to the correct size
        preprocess = self._fingerprinter.preprocess
        initial_guess = preprocess.inverse(from_centres)

        builder = None
        if atomic_numbers:
            # Fix the atomic numbers
            builder = atomic.AtomsCollectionBuilder(num_atoms)
            builder.numbers = atomic_numbers

        return self._optimiser.optimise(phi, initial_guess, atoms_builder=builder)


def get_best_rms(reference: atomic.AtomsCollection, probe: atomic.AtomsCollection) -> float:
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
    reorder_map = rmsdlib.reorder_hungarian(reference.numbers, probe.numbers, reference.positions, probe.positions)
    reordered = atomic.AtomsCollection(
        reference.num_atoms,
        positions=probe.positions[reorder_map],
        numbers=probe.numbers[reorder_map],
    )

    # Now ask rdkit to find the best.
    # This will try a brute-force permutation search starting with the one we just defined which is likely to be
    # the best
    reference = rdkittools.milad2rdkit(reference)
    probe = rdkittools.milad2rdkit(reordered)
    return rdMolAlign.GetBestRMS(probe, reference)
