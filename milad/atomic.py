# -*- coding: utf-8 -*-
import logging
import numbers
from typing import Dict, Optional, Type, Tuple, Union

import numpy as np

from . import functions

_LOGGER = logging.getLogger(__name__)


class AtomsCollection(functions.PlainState):
    """A collection of atomic positions along with a numbers for each of their species.  Stored as:

    [x_1, y_1, z_1, x_2, y_2, z_2, ..., x_N, y_N, z_N, s_1, s_2, ..., S_N]

    where s_i are the numbers labelling the species
    """

    @staticmethod
    def contains_num_atoms(state: functions.StateLike) -> int:
        length = len(state)
        if (length % 4) != 0:
            raise ValueError('AtomsCollection state vectors must be a multiple of 4')
        return int(length / 4)

    @staticmethod
    def total_length(num_atoms: int) -> int:
        return 4 * num_atoms

    def __init__(self, num: int, positions: np.array = None, species: np.array = None):
        super().__init__(self.total_length(num))
        self._num_atoms = num
        if positions is not None:
            self.positions = positions
        if species is not None:
            self.numbers = species

    def copy(self) -> 'AtomsCollection':
        atoms = AtomsCollection(self._num_atoms)
        np.copyto(atoms.vector, self.vector)
        return atoms

    def linear_pos_idx(self, atom_idx) -> slice:
        """Get the slice of the linear vector containing the position of the atom with the given
        index"""
        if atom_idx < 0 or atom_idx >= self._num_atoms:
            raise IndexError(atom_idx)
        return slice(3 * atom_idx, 3 * (atom_idx + 1))

    def linear_number_idx(self, atom_idx) -> int:
        """Get the index of the weight of the atom with the given index"""
        if atom_idx < 0 or atom_idx >= self._num_atoms:
            raise IndexError(atom_idx)

        return 3 * self._num_atoms + atom_idx

    @property
    def num_atoms(self):
        return self._num_atoms

    @property
    def positions(self) -> np.array:
        positions = self.vector[:3 * self._num_atoms]
        positions.shape = (self._num_atoms, 3)
        return positions

    @positions.setter
    def positions(self, new_positions: np.array):
        positions = self.vector[:3 * self._num_atoms]
        positions.shape = (self._num_atoms, 3)
        positions[:, :] = new_positions

    @property
    def numbers(self) -> np.array:
        return self.vector[3 * self._num_atoms:]

    @numbers.setter
    def numbers(self, new_numbers: np.array):
        self._vector[3 * self._num_atoms:] = new_numbers


class AtomsCollectionBuilder(functions.Function):
    """Take a vector as an input and build the corresponding atoms collection from it"""
    input_type = np.ndarray
    output_type = AtomsCollection
    supports_jacobian = True

    def __init__(self, num_atoms: int):
        super().__init__()
        self._num_atoms = num_atoms
        self._mask = np.empty(4 * num_atoms, dtype=object)
        self._mask.fill(None)

    @property
    def positions(self) -> np.ndarray:
        positions = self._mask[:3 * self._num_atoms]
        positions.shape = (self._num_atoms, 3)
        return positions

    @positions.setter
    def positions(self, value: np.ndarray):
        self._mask[:3 * self._num_atoms] = value.reshape(3 * self._num_atoms)

    @property
    def numbers(self) -> np.array:
        return self._mask[3 * self._num_atoms:]

    @numbers.setter
    def numbers(self, value: np.ndarray):
        self._mask[3 * self._num_atoms:] = value

    @property
    def input_length(self) -> int:
        return (self._mask == None).sum()  # pylint: disable=singleton-comparison

    @property
    def inverse(self) -> 'AtomsCollectionBuilder.Inverse':
        return AtomsCollectionBuilder.Inverse(self._num_atoms, self._mask)

    def empty_output(self, in_state: functions.State) -> functions.State:
        return AtomsCollection(self._num_atoms)

    def output_length(self, in_state: functions.State) -> int:
        return 4 * self._num_atoms

    def evaluate(self, state: functions.State, get_jacobian=False) -> AtomsCollection:
        vector = functions.get_bare_vector(state)
        if len(vector) != self.input_length:
            raise ValueError(f'Expected input of length {self.input_length} but got {len(vector)}')

        atoms_collection = AtomsCollection(self._num_atoms)
        fixed_indexes = np.where(self._mask != None)  # pylint: disable=singleton-comparison
        atoms_collection.vector[fixed_indexes] = self._mask[fixed_indexes]
        # Copy over from the new vector
        atoms_collection.vector[np.where(self._mask == None)[0]] = vector  # pylint: disable=singleton-comparison

        if get_jacobian:
            jacobian = np.zeros((self.output_length(state), len(state)))
            input_idx = 0

            for idx, entry in enumerate(self._mask):
                if entry is None:
                    jacobian[idx, input_idx] = 1.
                    input_idx += 1

            return atoms_collection, jacobian

        return atoms_collection

    class Inverse(functions.Function):
        input_type = AtomsCollection
        supports_jacobian = False

        def __init__(self, num_atoms: int, mask):
            super().__init__()
            self._num_atoms = num_atoms
            self._mask = mask

        @property
        def inverse(self) -> 'AtomsCollectionBuilder':
            builder = AtomsCollectionBuilder(self._num_atoms)
            builder._mask = self._mask
            return builder

        def output_length(self, in_state: AtomsCollection) -> int:
            return (self._mask == None).sum()

        def evaluate(self, state: AtomsCollection, get_jacobian=False) -> np.ndarray:
            out_vector = np.empty(self.output_length(state))
            output_length = self.output_length(state)
            if len(out_vector) != output_length:
                raise ValueError(f'Expected input of length {output_length} but got {state.vector.size}')

            current_idx = 0
            for idx, mask in enumerate(self._mask):
                if mask is None:
                    out_vector[current_idx] = state.vector[idx]
                    current_idx += 1

            return out_vector


class AtomsCollectionBuilder2:
    """An atoms collection builder that doesn't require a fixed number of atoms to be defined"""
    input_type = np.ndarray
    output_type = AtomsCollection
    supports_jacobian = True

    def __init__(self, species=True):
        super().__init__()
        self._species = species

    def evaluate(self,
                 state: functions.State,
                 get_jacobian=False) -> Union[AtomsCollection, Tuple[AtomsCollection, np.ndarray]]:
        vector = functions.get_bare_vector(state)
        length = len(vector)
        if (length % 4) != 0:
            raise ValueError(f'Passed vector is not a multiple of 4 ({length})')

        atoms = AtomsCollection(length / 4)
        atoms.vector[:] = vector

        if get_jacobian:
            return atoms, np.identity(len(vector))

        return atoms


class FeatureMapper(functions.Function):
    """Map a collection of atoms onto a generic feature type such as Delta functions or Gaussians"""
    input_type = AtomsCollection
    output_type = functions.Features
    supports_jacobian = True

    def __init__(self, feature: Type[functions.Feature], feature_kwargs=None, map_species_to: int = None):
        super().__init__()
        self._feature_type = feature
        self._map_species_to = map_species_to
        self._feature_kwargs = feature_kwargs or {}
        self._inverse = self.Inverse(self)

    @property
    def inverse(self):
        return self._inverse

    @property
    def map_species_to(self) -> Optional[str]:
        return self._map_species_to

    def empty_output(self, in_state: AtomsCollection) -> functions.Features:
        return functions.Features()

    def output_length(self, in_state: AtomsCollection) -> int:
        return in_state.num_atoms * self._feature_type.LENGTH

    def evaluate(self, atoms: AtomsCollection, get_jacobian=False) -> functions.Features:
        features = functions.Features()

        jac = None if not get_jacobian else np.zeros((self.output_length(atoms), len(atoms)))

        # Create the features one by one and add them to the features vector
        idx = 0  # Keep track of where we are in the features vector
        for atom_idx, (pos, specie) in enumerate(zip(atoms.positions, atoms.numbers)):
            kwargs = self._feature_kwargs.copy()
            kwargs['pos'] = pos
            feature = self._feature_type(**kwargs)
            if self._map_species_to:
                feature.vector[self._map_species_to] = specie

            if get_jacobian:
                # The structure of each feature is assumed to be [x, y, z, ..., s, ...]
                # where we optionally map the species to the index s
                pos_idx = atoms.linear_pos_idx(atom_idx).start
                for i in range(0, 3):
                    jac[idx + i, pos_idx + i] = 1.

                if self._map_species_to:
                    jac[idx + self._map_species_to, atoms.linear_number_idx(atom_idx)] = 1.

            features.add(feature)
            idx += len(feature)

        if get_jacobian:
            return features, jac

        return features

    class Inverse(functions.Function):
        input_type = functions.Features
        output_type = AtomsCollection
        supports_jacobian = False

        def __init__(self, mapper: 'FeatureMapper'):
            super().__init__()
            self._mapper = mapper

        def output_length(self, features: functions.Features) -> int:
            return AtomsCollection.total_length(len(features.features))

        def evaluate(self, features: functions.Features, get_jacobian=False):
            atoms_collection = AtomsCollection(len(features.features))

            for idx, feature in enumerate(features.features):
                atoms_collection.positions[idx] = feature.pos
                if self._mapper.map_species_to:
                    atoms_collection.numbers[idx] = getattr(feature, self._mapper.map_species_to)

            return atoms_collection


class ScalePositions(functions.Function):
    """Scale the atomic system by a fixed scale factor"""

    input_type = AtomsCollection
    output_type = AtomsCollection
    supports_jacobian = True

    def __init__(self, scale_factor: float):
        super().__init__()
        self._scale_factor = scale_factor

    def empty_output(self, in_state: AtomsCollection) -> AtomsCollection:
        return AtomsCollection(in_state.num_atoms)

    @property
    def inverse(self) -> Optional[functions.Function]:
        return ScalePositions(1 / self._scale_factor)

    def output_length(self, in_state: AtomsCollection) -> int:
        return len(in_state)

    def evaluate(self, atoms: AtomsCollection, get_jacobian=False):
        out_atoms = AtomsCollection(atoms.num_atoms)
        out_atoms.positions[:] = self._scale_factor * atoms.positions
        out_atoms.numbers[:] = atoms.numbers

        if get_jacobian:
            jac = np.zeros((self.output_length(atoms), len(atoms)))
            final_pos_idx = 3 * atoms.num_atoms
            np.fill_diagonal(jac[0:final_pos_idx, 0:final_pos_idx], self._scale_factor)
            np.fill_diagonal(jac[final_pos_idx:, final_pos_idx:], 1.)
            return out_atoms, jac

        return out_atoms


class CentreAtomsCollection(functions.Function):
    """Centre an atomic system by translating it to the centre of mass"""

    input_type = AtomsCollection
    output_type = AtomsCollection
    supports_jacobian = False

    def output_length(self, in_state: AtomsCollection) -> int:
        return len(in_state)

    def evaluate(self, in_atoms: AtomsCollection, get_jacobian=False) -> AtomsCollection:
        out_atoms = AtomsCollection(in_atoms.num_atoms)

        centre = np.sum(in_atoms.positions) / in_atoms.num_atoms
        out_atoms.positions = in_atoms.positions - centre
        out_atoms.numbers = in_atoms.numbers

        return out_atoms


class MapNumbers(functions.Function):
    """Map atom the given set of atom numbers onto a continuous range"""
    input_type = AtomsCollection
    output_type = AtomsCollection
    supports_jacobian = False

    def __init__(self, possible_numbers: set, mapped_range: Tuple[float, float]):
        super().__init__()
        self._numbers = list(sorted(possible_numbers))
        self._mapped_range = mapped_range
        self._range_size = mapped_range[1] - mapped_range[0]
        self._half_bin = self._range_size / (2 * len(self._numbers))

    @property
    def inverse(self) -> Optional['functions.Function']:
        return MapNumbers.Inverse(set(self._numbers), self._mapped_range)

    def empty_output(self, in_state: AtomsCollection) -> AtomsCollection:
        return AtomsCollection(in_state.num_atoms)

    def output_length(self, in_state: AtomsCollection) -> int:
        return len(in_state)

    def evaluate(self, in_atoms: AtomsCollection, get_jacobian=False) -> AtomsCollection:
        out_atoms = in_atoms.copy()

        # Now adjust the numbers
        for idx, num in enumerate(out_atoms.numbers):
            try:
                num_idx = self._numbers.index(num)
            except ValueError:
                pass
            else:
                new_numbers = (num_idx / len(self._numbers)) * self._range_size + self._mapped_range[0]
                out_atoms.numbers[idx] = new_numbers + self._half_bin

        return out_atoms

    class Inverse(functions.Function):

        def __init__(self, possible_numbers: set, mapped_range: Tuple[float, float]):
            super().__init__()
            self._numbers = list(sorted(possible_numbers))
            self._mapped_range = mapped_range
            self._range_size = mapped_range[1] - mapped_range[0]

        def output_length(self, in_state: AtomsCollection) -> int:
            return len(in_state)

        def empty_output(self, in_state: AtomsCollection) -> AtomsCollection:
            return AtomsCollection(in_state.num_atoms)

        def evaluate(self, in_atoms: AtomsCollection, get_jacobian=False) -> AtomsCollection:
            out_atoms = in_atoms.copy()

            # Now adjust the numbers
            for idx, num in enumerate(out_atoms.numbers):
                if num >= self._mapped_range[0] and num <= self._mapped_range[1]:
                    rescaled = (num - self._mapped_range[0]) / self._range_size * len(self._numbers)
                    out_atoms.numbers[idx] = self._numbers[int(rescaled)]
                else:
                    _LOGGER.warning(
                        'Got a species number that is no in the range: %n <= %n <= %n', self._mapped_range[0], num,
                        self._mapped_range[1]
                    )

            return out_atoms


class ApplyCutoff(functions.Function):
    """Given a collection of atoms exclude any that are further from the origin than the given cutoff"""

    def __init__(self, cutoff: float):
        super().__init__()
        self._cutoff_sq = cutoff * cutoff

    def evaluate(self, atoms: AtomsCollection, get_jacobian=False):
        index_map = {}
        # Find all those that are within the cutoff
        for idx in range(atoms.num_atoms):
            pos = atoms.positions[idx]
            if np.dot(pos, pos) < self._cutoff_sq:
                index_map[idx] = len(index_map)

        transformed = AtomsCollection(
            len(index_map),
            positions=atoms.positions[tuple(index_map.keys()), :],
            species=atoms.numbers[(tuple(index_map.keys()),)]
        )

        if get_jacobian:
            jac = np.zeros(len(index_map), len(atoms))
            for old_idx, new_idx in index_map.items():
                jac[atoms.linear_pos_idx(old_idx), transformed.linear_pos_idx(new_idx)] = 1.
                jac[atoms.linear_number_idx(old_idx), transformed.linear_number_idx(new_idx)] = 1.

            return atoms, jac

        return atoms