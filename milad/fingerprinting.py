# -*- coding: utf-8 -*-
from typing import Optional, Type, Tuple, Union

import numpy as np

from . import atomic
from . import base_moments
from . import invariants
from . import functions
from . import zernike

__all__ = 'MomentInvariantsDescriptors', 'descriptor', 'Fingerprinter', 'fingerprinter'


class MomentInvariantsDescriptors(functions.Function):
    """Class that is responsible for producing fingerprints form atomic environments"""

    def __init__(
        self,
        feature_mapper: atomic.FeatureMapper,
        cutoff: float = None,
        moments_calculator=None,
        invs: invariants.MomentInvariants = None,
        preprocess: functions.Function = None
    ):
        super().__init__()
        preprocess = preprocess or functions.Identity()

        # Now the actual fingerprinting
        process = functions.Chain(feature_mapper)
        self._invariants = invs or invariants.read(invariants.COMPLEX_INVARIANTS)
        moments_calculator = moments_calculator or zernike.ZernikeMomentCalculator(self._invariants.max_order)

        process.append(moments_calculator)
        process.append(self._invariants)

        self._cutoff = cutoff
        self._preprocess = preprocess
        self._process = process
        # Combine the two steps in one calculator
        self._calculator = functions.Chain(preprocess, process)

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
    moments_calculator=None,
    invs: invariants.MomentInvariants = None,
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
    :param scale:
    :param moments_calculator:
    :param invs:
    :return:
    """
    # Set up the preprocessing
    preprocess = functions.Chain()

    species = species or {}

    species_map = species.get('map', {})
    if species_map:
        preprocess.append(atomic.MapNumbers(species=species_map['numbers'], map_to=species_map['range']))

    if cutoff is not None:
        preprocess.append(atomic.ApplyCutoff(cutoff))

        if scale:
            # Rescale everything to be in the range [-1, 1], the typical domain of orthogonality
            preprocess.append(atomic.ScalePositions(1. / cutoff))

    features = features or dict(type=functions.WeightedDelta, map_species_to=species_map.get('to', None))
    return MomentInvariantsDescriptors(
        feature_mapper=atomic.FeatureMapper(**features),
        cutoff=cutoff,
        moments_calculator=moments_calculator,
        invs=invs,
        preprocess=preprocess
    )


# Alias for backwards compatibility
Fingerprinter = MomentInvariantsDescriptors
fingerprinter = descriptor
