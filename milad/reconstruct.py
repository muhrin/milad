# -*- coding: utf-8 -*-
from typing import Union, Type, Tuple, Optional

import numpy as np
from scipy import optimize

from . import atomic
from . import functions
from . import invariants
from . import zernike


class Encoder(functions.Function):
    supports_jacobian = False

    def __init__(
        self,
        preprocess: functions.Chain,
        encode: functions.Chain,
    ):
        super().__init__()
        self._preprocess = preprocess
        self._encode = encode

    @property
    def preprocess(self) -> functions.Chain:
        return self._preprocess

    @property
    def encode(self) -> functions.Function:
        return self._encode

    def evaluate(self, state: functions.StateLike, get_jacobian=False):
        preprocessed = self._preprocess(state)
        return self._encode(preprocessed, get_jacobian)

    def decoder(self, species_range: Tuple[float, float], initial_structure: atomic.AtomsCollection):
        return Decoder(self, species_range, initial_structure)


class Decoder(functions.Function):
    supports_jacobian = False

    def __init__(
        self, enc: Encoder, species_range: Tuple[float, float], starting_config: atomic.AtomsCollection, xtol=1e-5
    ):
        super().__init__()
        self._encoder = enc
        self._num_atoms = starting_config.num_atoms
        self._species_range = species_range
        self._starting_config = starting_config
        self._xtol = xtol
        self._lower_bounds, self._upper_bounds = self._calculate_bounds()

    def evaluate(self, target: functions.StateLike, get_jacobian=False):
        residuals = functions.Chain(*self._encoder.encode._functions)
        residuals.append(functions.Residuals(target))

        previous_result = None

        def calc(state: functions.StateLike):
            global previous_result
            res, jac = residuals(state, jacobian=True)
            print(f'{np.abs(res).max()}')

            previous_result = state, jac.real
            return res.real

        def jac(state: functions.StateLike):
            global previous_result
            if np.all(previous_result[0] == state):
                return previous_result[1]

            _, jacobian = residuals(state, jacobian=True)
            return jacobian.real

        x0 = self._get_starting_config()

        result = optimize.least_squares(
            calc,
            x0=x0,
            jac=jac,
            bounds=(self._lower_bounds, self._upper_bounds),
            xtol=self._xtol,
        )

        if not result.success:
            raise RuntimeError(f'Could not decode the passed input: {result.message}')

        return self._encoder.preprocess.inverse(result.x)

    def _get_starting_config(self):
        return self._encoder.preprocess(self._starting_config)

    def _calculate_bounds(self):
        prepare = self._encoder.preprocess[-1]
        atoms = atomic.AtomsCollection(self._num_atoms)

        # Lower
        atoms.positions = -1
        atoms.numbers = self._species_range[0]
        lower = prepare(atoms)

        # Upper
        atoms.positions = 1
        atoms.numbers = self._species_range[1]
        upper = prepare(atoms)

        return lower, upper


def encoder(
    num_atoms: int,
    allowed_species: set,
    species_number_range=(0.5, 5.),
    scale_factor=None,
    feature_type=functions.WeightedDelta,
    feature_kwargs=None,
    map_species_to='WEIGHT',
    moments_calculator=None,
    invs: invariants.MomentInvariants = None,
):
    builder = atomic.AtomsCollectionBuilder(num_atoms)
    preprocess = functions.Chain(
        atomic.MapNumbers(allowed_species, species_number_range), atomic.ScalePositions(scale_factor), builder.inverse
    )

    encode = functions.Chain(builder)
    encode.append(
        atomic.FeatureMapper(
            feature_type,
            feature_kwargs=feature_kwargs,
            map_species_to=get_species_map_idx(feature_type, map_species_to)
        )
    )
    invs = invs or invariants.read(invariants.COMPLEX_INVARIANTS)
    moments_calculator = moments_calculator or zernike.ZernikeMomentCalculator(invs.max_order)

    encode.append(moments_calculator)
    encode.append(invs)

    return Encoder(preprocess, encode)


class Fingerprinter(functions.Function):

    def __init__(
        self,
        species: set,
        species_number_range=(0.5, 5.),
        cutoff: float = None,
        feature_type=functions.WeightedDelta,
        feature_kwargs=None,
        map_species_to='WEIGHT',
        moments_calculator=None,
        invs: invariants.MomentInvariants = None,
    ):
        super().__init__()
        fingerprint = functions.Chain(atomic.MapNumbers(species, species_number_range))
        if cutoff is not None:
            fingerprint.append(atomic.ScalePositions(1. / cutoff))

        # The descriptor is typically defined in the domain [-1,1] to apply the cutoff
        fingerprint.append(atomic.ApplyCutoff(1.))
        fingerprint.append(
            atomic.FeatureMapper(
                feature_type,
                feature_kwargs=feature_kwargs,
                map_species_to=get_species_map_idx(feature_type, map_species_to)
            ),
        )
        invs = invs or invariants.read(invariants.COMPLEX_INVARIANTS)
        moments_calculator = moments_calculator or zernike.ZernikeMomentCalculator(invs.max_order)

        fingerprint.append(moments_calculator)
        fingerprint.append(invs)

        self._fingerprint = fingerprint

    def evaluate(self, state: atomic.AtomsCollection, get_jacobian=False):
        result = self._fingerprint(state, get_jacobian)
        if get_jacobian:
            return result[0].real, result[1].real

        return result.real

    def atom_centred(self, atoms: atomic.AtomsCollection, idx: int, get_jacobian=False):
        new_centre = atoms.positions[idx]
        new_atoms = atomic.AtomsCollection(
            atoms.num_atoms, positions=atoms.positions - new_centre, species=atoms.numbers
        )
        return self.evaluate(new_atoms, get_jacobian=get_jacobian)


def get_species_map_idx(feature_type: Type[functions.Feature], map_to: Union[int, str]) -> Optional[int]:
    if isinstance(map_to, int):
        return map_to
    if isinstance(map_to, str):
        return getattr(feature_type, map_to)
    if map_to is None:
        return None

    raise TypeError(map_to)
