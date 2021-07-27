# -*- coding: utf-8 -*-
import functools
import collections
import logging
from typing import List

import numpy as np
import rmsd as rmsdlib
from scipy.spatial import distance
from scipy import optimize
from sklearn import cluster

from . import atomic
from . import base_moments
from . import exceptions
from . import fingerprinting
from . import mathutil
from . import optimisers

_LOGGER = logging.getLogger(__name__)

StructureOptimisationResult = collections.namedtuple('StructureOptimisationResult', 'success atoms message rmsd')


@functools.singledispatch
def find_clusters(spec, num_clusters: int, **kwargs) -> np.ndarray:
    """Find clusters based on moments"""
    raise TypeError(f'Cannot find clusters from {spec.__class__.__name__}')


@find_clusters.register(base_moments.Moments)
def _(moments: base_moments.Moments, num_clusters: int, query: base_moments.ReconstructionQuery,
      descriptor: fingerprinting.MomentInvariantsDescriptor) -> np.ndarray:
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


def get_surrounding_gridpoints(
    query: base_moments.ReconstructionQuery, atom_pos: np.array, radius: float
) -> np.ndarray:
    # First find the points within a cube of the central point
    mask = (query.points[:, 0] > atom_pos[0] - radius) & \
           (query.points[:, 0] < atom_pos[0] + radius) & \
           (query.points[:, 1] > atom_pos[1] - radius) & \
           (query.points[:, 1] < atom_pos[1] + radius) & \
           (query.points[:, 2] > atom_pos[2] - radius) & \
           (query.points[:, 2] < atom_pos[2] + radius)

    indices = np.argwhere(mask)[:, 0]
    rsq = radius * radius

    # Now check the gridpoints left to see if they fall within the cutoff sphere
    for idx in indices:
        dr = query.points[idx] - atom_pos  # pylint: disable=invalid-name
        if np.dot(dr, dr) > rsq:
            # Remove it from the mask
            mask[idx] = False

    return mask


def get_buffer_indices(query: base_moments.ReconstructionQuery, grid_values) -> List[int]:
    # Get all grid values that are currently 0, these should stay zero
    outside = list(np.argwhere(grid_values == 0.).reshape(-1))

    # Now, get those that are near the bounds of the sphere i.e. 1.0
    cutoff_sq = 0.85**2
    for idx, pt in enumerate(query.points):  # pylint: disable=invalid-name
        if np.sum(pt**2) > cutoff_sq:
            outside.append(idx)

    return outside


def find_atoms(
    # pylint: disable=too-many-locals
    num_atoms: int,
    moments: base_moments.Moments,
    descriptor: fingerprinting.MomentInvariantsDescriptor,
    query: base_moments.ReconstructionQuery = None,
    grid_size=31,
) -> atomic.AtomsCollection:
    """
    Given a set of moments this will try to extract atom positions and species by successively placing a new atom
    on the current peak in reconstruction grid.  After this the a new grid is calculated and subtracted from the
    original.

    :param moments:
    :param descriptor:
    :param num_atoms: the number of atoms to find
    :param query: the reconstruction query to use.
    :param grid_size: the size of reconstruction grid to use.  Only used if no reconstruction query is passed.
    :return:
    """
    query = query or moments.create_reconstruction_query(moments.get_grid(grid_size), moments.max_order)

    atom_positions = []
    atom_numbers = []
    optimiser = optimisers.StructureOptimiser()

    # Calculate the original moments grid
    orig_grid = moments.reconstruct(query, zero_outside_domain=True)
    current_grid = orig_grid

    preprocess = False

    while True:
        # Find the index of the maximum value in the current grid
        max_idx = current_grid.argmax()

        # Get that position in the grid
        atom_positions.append(query.points[max_idx] / descriptor.cutoff)
        atom_numbers.append(1.)

        # Build an atoms collection with the current set of atoms
        current_atoms = atomic.AtomsCollection(len(atom_positions), positions=atom_positions, numbers=atom_numbers)

        # Locally optimise the atomic positions
        res = optimiser.optimise(
            descriptor,
            target=moments,
            initial=current_atoms,
            preprocess=preprocess,
        )

        if num_atoms == current_atoms.num_atoms:
            # Reached the number of atoms limit
            break

        # Get the moments so we can subtract this from the grid
        current_moments = descriptor.get_moments(res.value, preprocess=preprocess)

        # Get the grid for this set of atoms
        this_grid = current_moments.reconstruct(query, zero_outside_domain=True)

        # Subtract off the current grid
        current_grid = orig_grid - this_grid

        # Now remove the signal of the atom from the grid
        remove_idxs = np.argwhere(current_grid >= (0.5 * current_grid.max()))
        current_grid[remove_idxs] = 0.

        atom_positions = list(res.value.positions)
        atom_numbers = list(res.value.numbers)

    return res.value


def create_atoms_collection(clusters: cluster.KMeans, atomic_numbers=1.):
    """Take a set of clusters and use the centres to construct an atoms collection"""
    num_atoms = len(clusters.cluster_centers_)
    return atomic.AtomsCollection(num_atoms, positions=clusters.cluster_centers_, numbers=atomic_numbers)


DecoderResult = collections.namedtuple(
    'DecoderResult', 'success message value rmsd moments_reconstruction initial_reconstruction atoms_reconstruction'
)


def merge_atoms(system: atomic.AtomsCollection, dist_threshold=0.2):
    # Proceed to merging of atoms
    dists = distance.cdist(system.positions, system.positions)
    np.fill_diagonal(dists, np.inf)  # Get rid of diagonals as this is just the self-interaction
    merge_sets = []
    for (i, j) in np.argwhere(dists < dist_threshold):
        if i > j:
            # Ignore lower diagonal
            continue

        merged = False
        for merge_set in merge_sets:
            if i in merge_set or j in merge_set:
                if i in merge_set:
                    merge_set.add(j)
                elif j in merge_set:
                    merge_set.add(i)
                merged = True
                break

        if not merged:
            # Create a new merge set
            merge_sets.append({i, j})

    # Start merging
    positions = []
    numbers = []
    merged_indices = set()
    for merge_set in merge_sets:
        pos = np.zeros(3)
        number = 0.
        for i in merge_set:
            pos += system.positions[i]
            number += system.numbers[i]
            merged_indices.add(i)
        pos /= len(merge_set)

        positions.append(pos)
        numbers.append(number)

    for i in set(range(len(system.numbers))) - merged_indices:
        positions.append(system.positions[i])
        numbers.append(system.numbers[i])

    return len(merged_indices), atomic.AtomsCollection(len(positions), positions, numbers)


def get_best_rms(
    reference: atomic.AtomsCollection,
    probe: atomic.AtomsCollection,
    max_attempts: int = 1000,
    max_retries=20,
    threshold=1e-7,
    use_hungarian=True
) -> float:
    """
    Get the best RMSs fitting between two molecules.  This will first use an algorithm to make a decent guess at the
    best permutational ordering of atoms and then try a brute force search.

    :param reference: the reference to fit to
    :param probe: the probe to try to fit to the reference
    :param max_attempts: the number of attempts that RDKit should make the find the best RMSD in each retry
    :param max_retries: the number of ties we should call RDKit's get best RMSD method
    :param threshold: if the RMSD returned by RDKit's get best RMSD method drops below this value then we stop and
        return the result
    :param use_hungarian: if True will use the Hungarian algorithm (https://en.wikipedia.org/wiki/Hungarian_algorithm)
        to try and find a good label assignment before calling RDKit
    :return: the best RMSD found
    """
    from rdkit.Chem import rdMolAlign
    from . import rdkittools

    if use_hungarian:
        # Find a decent re-ordering of the atoms
        try:
            reorder_map = rmsdlib.reorder_hungarian(
                reference.numbers, probe.numbers, reference.positions, probe.positions
            )
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


def find_iteratively(
    descriptor: fingerprinting.MomentInvariantsDescriptor,
    fingerprint: np.ndarray,
    num_atoms: int,
    initial: atomic.AtomsCollection,
    find_species=False,
    min_rmsd=1e-7,
    max_iters=6,
    grid_query=None,
    structure_optimiser=None,
    minsep=0.9,
    verbose=False,
):
    # pylint: disable=too-many-locals, too-many-branches
    # Initialisation
    moments_optimiser = optimisers.MomentsOptimiser()
    if structure_optimiser is None:
        structure_optimiser = optimisers.StructureOptimiser()
        if minsep:
            structure_optimiser.separation_force = atomic.SeparationForce(epsilon=1e-8, cutoff=minsep, power=6)

    atoms = initial
    mask = None
    if not find_species:
        # Fix the species numbers
        mask = atoms.get_mask()
        mask.numbers = atoms.numbers

    for i in range(max_iters):
        # Now recreate the moments from the atoms
        moments = descriptor.get_moments(atoms)

        if grid_query is None:
            # Create the reconstruction query the first time
            grid_query = moments.create_reconstruction_query(moments.get_grid(31), moments.max_order)

        if verbose:
            print(f'{i}: Finding moments from fingerprint...', end='')
        result = moments_optimiser.optimise(
            invariants_fn=descriptor.invariants,
            target=fingerprint,
            initial=moments,
            verbose=(verbose == 'high'),
        )
        moments = result.value
        if verbose:
            print(f'rmsd {result.rmsd}')

        if verbose:
            print(f'{i}: Finding atoms from moments...', end='')
        result = find_atoms_from_moments(
            descriptor,
            moments,
            num_atoms,
            minsep=minsep,
            mask=mask,
            grid_query=grid_query,
            structure_optimiser=structure_optimiser,
            verbose=(verbose == 'high')
        )
        atoms = result.value
        if verbose:
            print(f'rmsd {result.rmsd}')

        if verbose:
            print(f'{i}: Finding atoms from fingerprint...', end='')
        result = structure_optimiser.optimise(
            descriptor,
            target=fingerprint,
            initial=atoms,
            mask=mask,
            verbose=(verbose == 'high'),
        )
        if verbose:
            print(f'rmsd {result.rmsd}')

        if find_species:
            # Take the current result, fix the species and allow positions to vary
            atoms = result.value

            pos_mask = atoms.get_mask()
            pos_mask.numbers = atoms.numbers

            if verbose:
                print(f'{i}: Optimising atomic positions wrt fingerprint...', end='')
            result = structure_optimiser.optimise(
                descriptor,
                target=fingerprint,
                initial=atoms,
                mask=pos_mask,
                verbose=(verbose == 'high'),
            )
            if verbose:
                print(f'rmsd {result.rmsd}')

        if result.rmsd < min_rmsd:
            break

        atoms = result.value

    return result


# Interface for atoms finder:
# (descriptor, moments, num_atoms: int, *, verbose: Bool)


def find_atoms_from_moments(
    descriptor: fingerprinting.MomentInvariantsDescriptor,
    moments,
    num_atoms: int,
    *,
    numbers=1.,
    mask=None,
    grid_query=None,
    structure_optimiser=None,
    minsep=0.9,
    verbose=False,
):
    # Find the peaks and create the corresponding collection of atoms
    if verbose:
        print('Finding peaks...', end='')
    peaks = find_peaks(descriptor, moments, num_atoms, query=grid_query, subtract_signal=True, exclude_radius=minsep)
    atoms = atomic.AtomsCollection(num_atoms, peaks, numbers=numbers)
    if verbose:
        print(f'found {atoms.num_atoms}')

    optimiser = structure_optimiser or optimisers.StructureOptimiser()
    result = optimiser.optimise(
        descriptor,
        target=moments,
        initial=atoms,
        mask=mask,
        verbose=verbose,
    )

    return result


def find_maximum(
    descriptor: fingerprinting.MomentInvariantsDescriptor,
    moments: base_moments.Moments,
    num_samples=128,
) -> optimize.OptimizeResult:
    """Given a set of moments this function will use a function minimisation algorithm to find
    the global maximum of the moments reconstruction

    :param descriptor: the descriptor that created the moments
    :param moments: the moments
    :param num_samples: the number of samples to use for the first pass of the global maximum search
    """
    # pylint: disable=invalid-name
    cutoff = descriptor.cutoff

    def adapter(moms, pt):
        return -moms.value_at(np.array([mathutil.sph2cart(pt)]))

    res = optimize.shgo(
        lambda pt: adapter(moments, pt),
        bounds=[(0.0, 1.), (0., np.pi), (0., 2 * np.pi)],
        n=num_samples,
        sampling_method='sobol'
    )
    # Convert coordinates back to cartesian and scale
    res.update(dict(x=mathutil.sph2cart(res.x) / cutoff, xl=mathutil.sph2cart(res.x) / cutoff))
    return res


def find_peaks(
    descriptor: fingerprinting.MomentInvariantsDescriptor,
    moments: base_moments.Moments,
    num: int,
    *,
    query: base_moments.ReconstructionQuery = None,
    grid_size=31,
    exclude_radius=0.9,
    subtract_signal=True
):
    query = query or moments.create_reconstruction_query(moments.get_grid(grid_size), moments.max_order)
    current_grid = moments.reconstruct(query, zero_outside_domain=True)
    return find_peaks_from_grid(
        descriptor, num, current_grid, query, exclude_radius=exclude_radius, subtract_signal=subtract_signal
    )


def find_peaks_from_grid(
    descriptor: fingerprinting.MomentInvariantsDescriptor,
    num_peaks: int,
    grid_values: np.ndarray,
    query: base_moments.ReconstructionQuery,
    exclude_radius=0.95,
    subtract_signal=True
):
    # pylint: disable=too-many-locals
    scale = 1.0 if descriptor.cutoff is None else 1. / descriptor.cutoff

    current_grid = grid_values.copy()

    exclude = exclude_radius * scale
    outside = get_buffer_indices(query, current_grid)

    mask = np.zeros(current_grid.shape, dtype=bool)
    mask[outside] = True

    found_positions = []
    for _ in range(num_peaks):
        # Find the index of the maximum value in the current grid
        max_idx = current_grid.argmax()

        # Get that position in the grid
        atom_pos = query.points[max_idx]
        found_positions.append(atom_pos / scale)

        if subtract_signal:
            # Build an atoms collection with a single atom at that position
            single_atom = atomic.AtomsCollection(1, positions=[atom_pos / scale], numbers=[1.])

            # Get the moments so we can subtract this from the grid
            single_moments = descriptor.get_moments(single_atom, preprocess=False)

            # Get the grid for just that atom on its own
            atom_grid = single_moments.reconstruct(query, zero_outside_domain=True)

            # Subtract off the single atom grid
            current_grid -= atom_grid

        # Now mask off the gridpoints associated with this atom
        mask |= get_surrounding_gridpoints(query, atom_pos, exclude)

        current_grid[mask] = 0.

    return np.array(found_positions)
