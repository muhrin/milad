# -*- coding: utf-8 -*-
from typing import Optional, Type, Tuple, Union

import numpy as np

from . import atomic
from . import invariants
from . import functions
from . import zernike

__all__ = 'Fingerprinter', 'fingerprinter'


class Fingerprinter(functions.Function):
    """Class that is responsible for producing fingerprints form atomic environments"""

    def __init__(
        self,
        cutoff: float = None,
        feature_type=functions.WeightedDelta,
        feature_kwargs=None,
        map_species_to='WEIGHT',
        moments_calculator=None,
        invs: invariants.MomentInvariants = None,
        preprocess: functions.Function = None
    ):
        super().__init__()
        preprocess = preprocess or functions.Identity()

        # Now the actual fingerprinting
        process = functions.Chain()

        process.append(
            atomic.FeatureMapper(
                feature_type,
                feature_kwargs=feature_kwargs,
                map_species_to=get_species_map_idx(feature_type, map_species_to)
            ),
        )
        invs = invs or invariants.read(invariants.COMPLEX_INVARIANTS)
        moments_calculator = moments_calculator or zernike.ZernikeMomentCalculator(invs.max_order)

        process.append(moments_calculator)
        process.append(invs)

        self._cutoff = cutoff
        self._preprocess = preprocess
        self._process = process
        # Combine the two steps in one calculator
        self._calculator = functions.Chain(preprocess, process)

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

    def get_moments(self, atoms: atomic.AtomsCollection) -> np.ndarray:
        preprocessed = self.preprocess(atoms)
        return self.process[:-1](preprocessed)

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
            atoms.num_atoms, positions=atoms.positions - new_centre, species=atoms.numbers
        )
        return self.evaluate(new_atoms, get_jacobian=get_jacobian)


def fingerprinter(
    species: set,
    cutoff: float = None,
    scale=True,
    species_number_range=(0.5, 5.),
    map_species_to='WEIGHT',
    feature_type=functions.WeightedDelta,
    feature_kwargs=None,
    moments_calculator=None,
    invs: invariants.MomentInvariants = None,
):
    # Set up the preprocessing
    preprocess = functions.Chain()

    if map_species_to is not None:
        preprocess.append(atomic.MapNumbers(species, species_number_range))

    if cutoff is not None:
        preprocess.append(atomic.ApplyCutoff(cutoff))

        if scale:
            # Rescale everything to be in the range [-1, 1], the typical domain of orthogonality
            preprocess.append(atomic.ScalePositions(1. / cutoff))

    return Fingerprinter(
        cutoff=cutoff,
        feature_type=feature_type,
        feature_kwargs=feature_kwargs,
        map_species_to=map_species_to,
        moments_calculator=moments_calculator,
        invs=invs,
        preprocess=preprocess
    )


def get_species_map_idx(feature_type: Type[functions.Feature], map_to: Union[int, str]) -> Optional[int]:
    if isinstance(map_to, int):
        return map_to
    if isinstance(map_to, str):
        return getattr(feature_type, map_to)
    if map_to is None:
        return None

    raise TypeError(map_to)
