# -*- coding: utf-8 -*-
from typing import Optional, Tuple, List

import numpy as np

from . import atomic
from . import base_moments
from . import invariants
from . import functions
from . import zernike

__all__ = 'MomentInvariantsDescriptor', 'descriptor', 'Fingerprinter', 'fingerprinter'


class MomentInvariantsDescriptor(functions.Function):
    """Class that is responsible for producing fingerprints form atomic environments"""
    scaler = None

    def __init__(
        self,
        feature_mapper: atomic.FeatureMapper,
        moments_calculator: base_moments.MomentsCalculator,
        invs: invariants.MomentInvariants,
        cutoff: float = None,
        scale: bool = True,
        species_mapper: atomic.MapNumbers = None,
        apply_cutoff=True,
    ):
        super().__init__()

        # PREPROCESSING
        self._preprocess = functions.Chain()
        self._species_mapper = species_mapper
        if self._species_mapper is not None:
            self._preprocess.append(self._species_mapper)

        # Now the actual fingerprinting
        self._invariants = invs

        # PROCESSING
        # Create the actual fingerprinting process which is a chain of functions
        process = functions.Chain()
        if cutoff is not None:
            if apply_cutoff:
                process.append(atomic.ApplyCutoff(cutoff))

            if scale:
                # Rescale positions to be in the range |r| < 1, the typical domain of orthogonality
                self.scaler = atomic.ScalePositions(1. / cutoff)
                process.append(self.scaler)

        process.append(feature_mapper)
        process.append(moments_calculator)
        process.append(self._invariants)

        self._cutoff = cutoff
        self._process = process
        self._moments_calculator = moments_calculator
        # Combine the two steps in one calculator
        self._calculator = functions.Chain(self._preprocess, self._process)

    @property
    def invariants(self) -> invariants.MomentInvariants:
        return self._invariants

    @property
    def fingerprint_len(self) -> int:
        return len(self._invariants)

    @property
    def cutoff(self) -> Optional[float]:
        return self._cutoff

    @property
    def preprocess(self) -> functions.Chain:
        """Return the preprocessing function"""
        return self._preprocess

    @property
    def process(self) -> functions.Chain:
        """Return the processing function"""
        return self._process

    @property
    def moments_calculator(self) -> base_moments.MomentsCalculator:
        return self._moments_calculator

    @property
    def species(self) -> Optional[List[int]]:
        """Get the species (as integers) supported by this descriptor.  Returns None if there is no restriction"""
        if self._species_mapper is None:
            return None

        return self._species_mapper.numbers

    def get_moments(self, atoms: atomic.AtomsCollection, preprocess=True) -> base_moments.Moments:
        if preprocess:
            atoms = self.preprocess(atoms)
        return self.process[:-1](atoms)

    def evaluate(self, state: atomic.AtomsCollection, get_jacobian=False):
        result = self._calculator(state, get_jacobian)
        if get_jacobian:
            return result[0].real, result[1].real

        return result.real

    def fingerprint_and_derivatives(self, atoms: atomic.AtomsCollection) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method computes the fingerprint for the pass atoms collection and the corresponding position
        derivatives.
        :param atoms: the atoms collection to fingerprint
        :return: a tuple containing the fingerprint and the Jacobian
        """
        num_atoms = atoms.num_atoms

        # First preprocess as this may not be fully differentiable (and hence have no Jacobian)
        preprocessed = self.preprocess(atoms)

        # Now perform the rest of the fingerprinting procedure which does have derivatives
        fingerprint, jacobian = self.process(preprocessed, jacobian=True)
        len_fingerprint = len(fingerprint)

        # Now extract the portion of the Jacobian that relates just to atomic positions and reshape
        derivatives = jacobian[:, :3 * num_atoms].real
        # Reshape to be (len_fingerprint, num_atoms, 3) so xyz are stored in separate dimension
        derivatives = derivatives.reshape((len_fingerprint, num_atoms, 3))
        # Now sum xyz
        derivatives = derivatives.sum(axis=1)

        return fingerprint.real, derivatives

    def atom_centred(self, atoms: atomic.AtomsCollection, idx: int, get_jacobian=False):
        new_centre = atoms.positions[idx]
        new_atoms = atomic.AtomsCollection(
            atoms.num_atoms, positions=atoms.positions - new_centre, numbers=atoms.numbers
        )
        return self.evaluate(new_atoms, get_jacobian=get_jacobian)


def descriptor(
    features: Optional[dict] = None,
    species: Optional[dict] = None,
    cutoff: float = None,
    scale=True,
    moments_calculator: base_moments.MomentsCalculator = None,
    invs: invariants.MomentInvariants = None,
    apply_cutoff=True,
):
    """

    :param features:
    :param species: a dictionary that has the following form:
        {
            'map': {
                'numbers': Sequence[Number] - a sequence of atomic numbers to be mapped
                'range': Union[Number, Tuple[Number, Number]] - a number range to map species to
                'to': Union[int, str] - the feature value to map the numbers to e.g. 'WEIGHT' which is a class
                                            property of WeightedDelta
            }
        }
    :param cutoff:
    :param scale: if True scale the environments by a factor of 1 / cutoff to fit within typical orthogonality region
    :param moments_calculator: the moment calculator to use
    :param invs: the invariants to use
    :return:
    """
    # Set up the preprocessing
    species = species or {}
    species_map = species.get('map', {})
    if species_map:
        species_mapper = atomic.MapNumbers(species=species_map['numbers'], map_to=species_map['range'])
    else:
        species_mapper = None

    # Default to Zernike moments if not supplied
    invs = invs or invariants.read(invariants.COMPLEX_INVARIANTS)
    moments_calculator = moments_calculator or zernike.ZernikeMomentCalculator(invs.max_order)

    features = features or dict(type=functions.WeightedDelta, map_species_to=species_map.get('to', None))
    return MomentInvariantsDescriptor(
        feature_mapper=atomic.FeatureMapper(**features),
        moments_calculator=moments_calculator,
        invs=invs,
        cutoff=cutoff,
        scale=scale,
        species_mapper=species_mapper,
        apply_cutoff=apply_cutoff,
    )


# Aliases for backwards compatibility
Fingerprinter = MomentInvariantsDescriptor
fingerprinter = descriptor
