# -*- coding: utf-8 -*-
"""
Module containing functions and objects related to manipulating collections of atoms
"""
import logging
import random
from typing import Optional, Type, Tuple, Union, List

import numpy as np
import numpy.ma as ma
from scipy.spatial.distance import pdist

from . import functions
from . import generate

_LOGGER = logging.getLogger(__name__)

__all__ = "AtomsCollection", "random_atom_collection_in_sphere"


class AtomsCollection(functions.PlainState):
    """A collection of atomic positions along with a numbers for each of their species.  Stored as:

    [x_1, y_1, z_1, x_2, y_2, z_2, ..., x_N, y_N, z_N, s_1, s_2, ..., S_N]

    where s_i are the numbers labelling the species
    """

    @staticmethod
    def num_atomic_properties() -> int:
        """Returns the number of atomic properties i.e. variables that make up each atom.  Currently these are
        the x, y, z positions and the atomic specie but this may change."""
        return 4

    @staticmethod
    def total_length(num_atoms: int) -> int:
        return AtomsCollection.num_atomic_properties() * num_atoms

    def __init__(
        self, num: int, positions: np.array = None, numbers: np.array = None, dtype=None
    ):
        super().__init__(self.total_length(num), dtype=dtype)

        self._num_atoms = num
        if positions is not None:
            self.positions = positions
        if numbers is not None:
            self.numbers = numbers

    def __str__(self) -> str:
        return f"{len(self)} {str(self.numbers)}"

    def copy(self) -> "AtomsCollection":
        atoms = AtomsCollection(self._num_atoms, dtype=self.dtype)
        np.copyto(atoms.vector, self.vector)
        return atoms

    def linear_pos_idx(self, atom_idx: int) -> slice:
        """Get the slice of the linear vector containing the position of the atom with the given
        index"""
        if atom_idx < 0 or atom_idx >= self._num_atoms:
            raise IndexError(atom_idx)
        return slice(3 * atom_idx, 3 * (atom_idx + 1))

    def linear_number_idx(self, atom_idx: int) -> int:
        """Get the index of the weight of the atom with the given index"""
        if atom_idx < 0 or atom_idx >= self._num_atoms:
            raise IndexError(atom_idx)

        return 3 * self._num_atoms + atom_idx

    @property
    def num_atoms(self):
        return self._num_atoms

    @property
    def positions(self) -> np.array:
        positions = self.vector[: 3 * self._num_atoms]
        positions.shape = (self._num_atoms, 3)
        return positions

    @positions.setter
    def positions(self, new_positions: np.array):
        positions = self.vector[: 3 * self._num_atoms]
        positions.shape = (self._num_atoms, 3)
        positions[:, :] = new_positions

    @property
    def numbers(self) -> np.array:
        return self.vector[3 * self._num_atoms :]

    @numbers.setter
    def numbers(self, new_numbers: np.array):
        self._array[3 * self._num_atoms :] = new_numbers

    def get_mask(self, fill=None):
        """Get a mask for this atom collection.  Typically used during optimisation."""
        mask = AtomsCollection(self.num_atoms, dtype=object)
        mask.vector.fill(fill)
        return mask

    def get_builder(self, mask: "Optional[AtomsCollection]" = None):
        return AtomsCollectionBuilder(self._num_atoms, mask=mask)


class AtomsCollectionBuilder(functions.Function):
    """Take a vector as an input and build the corresponding atoms collection from it"""

    input_type = np.ndarray
    output_type = AtomsCollection
    supports_jacobian = True

    def __init__(self, num_atoms: int, mask: AtomsCollection = None):
        super().__init__()
        self._num_atoms = num_atoms
        if mask is None:
            self._mask = AtomsCollection(num_atoms, dtype=object)
            self._mask.positions[:] = None
            self._mask.numbers[:] = None
        else:
            self._mask = mask

    @property
    def num_atoms(self) -> int:
        """Get the number of atoms supported by this builder"""
        return self._num_atoms

    @property
    def mask(self) -> AtomsCollection:
        """Return the mask of fixed values (free values will have None entries)"""
        return self._mask

    @property
    def positions(self) -> np.ndarray:
        positions = self._mask[: 3 * self._num_atoms]
        positions.shape = (self._num_atoms, 3)
        return positions

    @positions.setter
    def positions(self, value: np.ndarray):
        self._mask[: 3 * self._num_atoms] = value.reshape(3 * self._num_atoms)

    @property
    def numbers(self) -> np.array:
        return self._mask[3 * self._num_atoms :]

    @numbers.setter
    def numbers(self, value: np.ndarray):
        self._mask[3 * self._num_atoms :] = value

    @property
    def input_length(self) -> int:
        return (self._mask.vector == None).sum()  # noqa: E711

    @property
    def inverse(self) -> "AtomsCollectionBuilder.Inverse":
        return AtomsCollectionBuilder.Inverse(self)

    def output_length(self, _in_state: functions.State) -> int:
        return 4 * self._num_atoms

    def apply_mask(self, atoms: AtomsCollection):
        """Given an atom collection this will set any values specified in the mask"""
        indices = tuple(
            np.argwhere(self._mask.vector != None).reshape(-1)  # noqa: E711
        )  # pylint: disable=singleton-comparison
        if len(indices) != 0:
            functions.copy_to(atoms.vector, indices, self._mask.vector, indices)

    def evaluate(
        self, state: functions.State, *, get_jacobian=False
    ) -> Union[AtomsCollection, Tuple[AtomsCollection, np.ndarray]]:
        vector = functions.get_bare(state)
        if len(vector) != self.input_length:
            raise ValueError(
                f"Expected input of length {self.input_length} but got {len(vector)}"
            )

        atoms = AtomsCollection(self._num_atoms)
        self.apply_mask(atoms)

        # Get the unmasked indices
        indices = np.argwhere(self._mask.vector == None).reshape(  # noqa: E711
            -1
        )  # pylint: disable=singleton-comparison
        functions.copy_to(atoms.vector, indices, vector)

        if get_jacobian:
            jacobian = np.zeros((self.output_length(state), len(state)))
            for in_idx, out_idx in enumerate(indices):
                jacobian[out_idx, in_idx] = 1.0

            return atoms, jacobian

        return atoms

    class Inverse(functions.Function):
        input_type = AtomsCollection
        supports_jacobian = False

        def __init__(self, builder: "AtomsCollectionBuilder"):
            super().__init__()
            self._builder = builder

        @property
        def inverse(self) -> "AtomsCollectionBuilder":
            return self._builder

        def output_length(self, _in_state: AtomsCollection) -> int:
            return self._builder.input_length

        def evaluate(
            # pylint: disable=unused-argument
            self,
            state: AtomsCollection,
            *,
            get_jacobian=False,
        ) -> np.ndarray:
            output_length = self.output_length(state)
            if self.output_length(state) != output_length:
                raise ValueError(
                    f"Expected input of length {output_length} but got {state.vector.size}"
                )

            indices = np.argwhere(
                self._builder.mask.vector == None  # noqa: E711
            )  # pylint: disable=singleton-comparison
            return state.vector[indices].reshape(-1)


class FeatureMapper(functions.Function):
    """Map a collection of atoms onto a generic feature type such as delta functions or Gaussians"""

    input_type = AtomsCollection
    output_type = functions.Features
    supports_jacobian = True

    def __init__(
        self,
        type: Type[
            functions.Feature
        ] = functions.WeightedDelta,  # pylint: disable=redefined-builtin
        kwargs: dict = None,
        map_species_to: Union[int, str] = None,
    ):
        """
        :param type: the feature type
        :param kwargs: dictionary of keyword arguments to pass to construct the features with
        :param map_species_to: optionally map atomic numbers to this index of the feature function vector
        """
        super().__init__()
        self._feature_type = type
        self._map_species_to = self._get_species_map_idx(type, map_species_to)
        self._feature_kwargs = kwargs or {}
        self._inverse = self.Inverse(self)

    def __repr__(self) -> str:
        desc = [self._feature_type.__name__]
        if self._feature_kwargs:
            desc.append(f"({self._feature_type})")
        desc.append(f"(species -> {self._map_species_to})")
        mapping = "".join(desc)
        return f"FeatureMapper({mapping})"

    @property
    def inverse(self):
        return self._inverse

    @property
    def map_species_to(self) -> Optional[int]:
        return self._map_species_to

    @staticmethod
    def _get_species_map_idx(
        feature_type: Type[functions.Feature], map_to: Union[int, str]
    ) -> Optional[int]:
        if map_to is None:
            return None

        map_idx = None
        if isinstance(map_to, int):
            map_idx = map_to
        elif isinstance(map_to, str):
            map_idx = getattr(feature_type, map_to)
        else:
            raise TypeError(map_to)

        if not isinstance(map_idx, int):
            raise TypeError(
                f"Expected '{map_to}' to correspond to an integer index, but got '{map_idx}'"
            )

        return map_idx

    def output_length(self, in_state: AtomsCollection) -> int:
        return in_state.num_atoms * self._feature_type.LENGTH

    def evaluate(
        # pylint: disable=arguments-differ
        self,
        atoms: AtomsCollection,
        *,
        get_jacobian=False,
    ) -> functions.Features:
        features = functions.Features()
        jac = (
            np.zeros((self.output_length(atoms), len(atoms))) if get_jacobian else None
        )

        # Create the features one by one and add them to the features vector
        idx = 0  # Keep track of where we are in the features vector
        for atom_idx, (pos, specie) in enumerate(zip(atoms.positions, atoms.numbers)):
            kwargs = self._feature_kwargs.copy()
            kwargs["pos"] = pos
            feature = self._feature_type(**kwargs)
            if self._map_species_to:
                feature.vector[self._map_species_to] = specie

            if get_jacobian:
                # The structure of each feature is assumed to be [x, y, z, ..., s, ...]
                # where we optionally map the species to the index s
                pos_idx = atoms.linear_pos_idx(atom_idx).start
                for i in range(0, 3):
                    jac[
                        idx + i, pos_idx + i
                    ] = 1.0  # pylint: disable=unsupported-assignment-operation

                if self._map_species_to:
                    # pylint: disable=unsupported-assignment-operation
                    jac[
                        idx + self._map_species_to, atoms.linear_number_idx(atom_idx)
                    ] = 1.0

            features.add(feature)
            idx += len(feature)

        if get_jacobian:
            return features, jac

        return features

    class Inverse(functions.Function):
        input_type = functions.Features
        output_type = AtomsCollection
        supports_jacobian = False

        def __init__(self, mapper: "FeatureMapper"):
            super().__init__()
            self._mapper = mapper

        @staticmethod
        def output_length(features: functions.Features) -> int:
            return AtomsCollection.total_length(len(features.features))

        def evaluate(
            # pylint: disable=unused-argument, arguments-differ
            self,
            features: functions.Features,
            get_jacobian=False,
        ):
            atoms_collection = AtomsCollection(len(features.features))

            for idx, feature in enumerate(features.features):
                atoms_collection.positions[idx] = feature.pos
                if self._mapper.map_species_to:
                    atoms_collection.numbers[idx] = feature.vector[
                        self._mapper.map_species_to
                    ]

            return atoms_collection


class ScalePositions(functions.Function):
    """Scale the atomic system by a fixed scale factor"""

    input_type = AtomsCollection
    output_type = AtomsCollection
    supports_jacobian = True

    def __init__(self, scale_factor: float):
        super().__init__()
        self._scale_factor = scale_factor

    @property
    def inverse(self) -> Optional[functions.Function]:
        return ScalePositions(1.0 / self._scale_factor)

    @staticmethod
    def output_length(in_state: AtomsCollection) -> int:
        return len(in_state)

    def evaluate(
        self, atoms: AtomsCollection, *, get_jacobian=False
    ):  # pylint: disable=arguments-differ
        out_atoms = AtomsCollection(atoms.num_atoms, dtype=atoms.dtype)
        out_atoms.positions[:] = self._scale_factor * atoms.positions
        out_atoms.numbers[:] = atoms.numbers

        if get_jacobian:
            jac = np.zeros((self.output_length(atoms), len(atoms)))
            final_pos_idx = 3 * atoms.num_atoms
            np.fill_diagonal(jac[0:final_pos_idx, 0:final_pos_idx], self._scale_factor)
            np.fill_diagonal(jac[final_pos_idx:, final_pos_idx:], 1.0)
            return out_atoms, jac

        return out_atoms


class CentreAtomsCollection(functions.Function):
    """Centre an atomic system by translating it such that the centroid is coincident with the origin"""

    input_type = AtomsCollection
    output_type = AtomsCollection
    supports_jacobian = False

    @property
    def inverse(self) -> Optional[functions.Function]:
        """It is not possible to 'uncenter' a set of atoms (as we dont' keep track of where the previous centre was)
        and so the inverse of this function simply does nothing."""
        return functions.Identity()

    @staticmethod
    def output_length(in_state: AtomsCollection) -> int:
        return len(in_state)

    def evaluate(
        # pylint: disable=unused-argument, arguments-differ
        self,
        in_atoms: AtomsCollection,
        *,
        get_jacobian=False,
    ) -> AtomsCollection:
        out_atoms = AtomsCollection(in_atoms.num_atoms)

        centre = np.sum(in_atoms.positions) / in_atoms.num_atoms
        out_atoms.positions = in_atoms.positions - centre
        out_atoms.numbers = in_atoms.numbers

        return out_atoms


class MapNumbers(functions.Function):
    """Map the given set of atom numbers onto a continuous range"""

    input_type = AtomsCollection
    output_type = AtomsCollection
    supports_jacobian = True

    def __init__(
        self, species: set, map_to: Union[float, Tuple[float, float]] = (1.0, 6.0)
    ):
        super().__init__()
        if not isinstance(map_to, tuple):
            # Assume it's a scalar and everything is being mapped to a single number
            map_to = (map_to, map_to)

        species = set(species)
        self._numbers = list(sorted(species))
        self._mapped_range = map_to
        self._range_size = map_to[1] - map_to[0]
        self._half_bin = self._range_size / (2 * len(self._numbers))

    @property
    def numbers(self) -> List[int]:
        return self._numbers

    @property
    def mapped_range(self) -> Tuple[float, float]:
        """Get the range that species are mapped onto"""
        return self._mapped_range

    @property
    def inverse(self) -> Optional["functions.Function"]:
        return MapNumbers.Inverse(set(self._numbers), self._mapped_range)

    @staticmethod
    def output_length(in_state: AtomsCollection) -> int:
        return len(in_state)

    def evaluate(
        # pylint: disable=unused-argument, arguments-differ
        self,
        in_atoms: AtomsCollection,
        get_jacobian=False,
    ) -> Union[AtomsCollection, Tuple[AtomsCollection, np.ndarray]]:
        out_atoms = in_atoms.copy()

        # Now adjust the numbers
        for idx, num in enumerate(out_atoms.numbers):
            try:
                num_idx = self._numbers.index(num)
            except ValueError:
                pass
            else:
                if num is not None:
                    new_numbers = (
                        num_idx / len(self._numbers)
                    ) * self._range_size + self._mapped_range[0]
                    out_atoms.numbers[idx] = new_numbers + self._half_bin

        if get_jacobian:
            # Create Jacobian that is identity apart from the species.
            # This part needs to be masked of as this mapping is not a continuous function
            natoms = in_atoms.num_atoms
            jac = ma.array(np.zeros((len(out_atoms), len(in_atoms))), mask=False)
            jac[: 3 * natoms, : 3 * natoms] = np.eye(natoms * 3)
            jac.mask[3 * natoms :, 3 * natoms :] = True

            return out_atoms, jac

        return out_atoms

    class Inverse(functions.Function):
        def __init__(self, possible_numbers: set, mapped_range: Tuple[float, float]):
            super().__init__()
            self._numbers = list(sorted(possible_numbers))
            self._mapped_range = mapped_range
            self._range_size = mapped_range[1] - mapped_range[0]

        @staticmethod
        def output_length(in_state: AtomsCollection) -> int:
            return len(in_state)

        def evaluate(
            # pylint: disable=unused-argument, arguments-differ
            self,
            in_atoms: AtomsCollection,
            *,
            get_jacobian=False,
        ) -> AtomsCollection:  # pylint: disable=arguments-differ
            out_atoms = in_atoms.copy()

            # Now adjust the numbers
            for idx, num in enumerate(out_atoms.numbers):
                if self._mapped_range[0] <= num <= self._mapped_range[1]:
                    if self._range_size == 0.0:
                        rescaled = self._mapped_range[0]
                    else:
                        rescaled = (
                            (num - self._mapped_range[0])
                            / self._range_size
                            * len(self._numbers)
                        )
                    out_atoms.numbers[idx] = self._numbers[int(rescaled)]
                else:
                    _LOGGER.warning(
                        "Got a species number that is no in the range: %i <= %i <= %i",
                        self._mapped_range[0],
                        num,
                        self._mapped_range[1],
                    )

            return out_atoms


class ApplyCutoff(functions.Function):
    """Given a collection of atoms exclude any that are further from the origin than the given cutoff"""

    input_type = AtomsCollection

    def __init__(self, cutoff: float):
        super().__init__()
        self._cutoff_sq = cutoff * cutoff

    @property
    def inverse(self) -> Optional[functions.Function]:
        """Naturally, this function is not fully invertible as atoms that have been cut off cannot be replaced the
        inverse is just the identity"""
        return functions.Identity()

    def evaluate(
        self, in_atoms: AtomsCollection, get_jacobian=False
    ):  # pylint: disable=arguments-differ
        index_map = {}
        # Find all those that are within the cutoff
        for idx in range(in_atoms.num_atoms):
            pos = in_atoms.positions[idx]
            if np.dot(pos, pos) < self._cutoff_sq:
                index_map[idx] = len(index_map)
            else:
                print("WARNING: CUTTING OFF {}".format(idx))

        out_atoms = AtomsCollection(
            len(index_map),
            positions=in_atoms.positions[tuple(index_map.keys()), :],
            numbers=in_atoms.numbers[(tuple(index_map.keys()),)],
        )

        if get_jacobian:
            jac = np.zeros((len(out_atoms), len(in_atoms)))
            for old_idx, new_idx in index_map.items():
                jac[
                    out_atoms.linear_pos_idx(new_idx), in_atoms.linear_pos_idx(old_idx)
                ] = np.eye(3)
                jac[
                    out_atoms.linear_number_idx(new_idx),
                    in_atoms.linear_number_idx(old_idx),
                ] = 1.0

            return out_atoms, jac

        return out_atoms


def random_atom_collection_in_sphere(
    num: int, radius=1.0, centre=True, numbers=1.0, minsep=None
) -> AtomsCollection:
    pts = generate.random_points_in_sphere(
        num, radius=radius, centre=centre, minsep=minsep
    )
    atoms = AtomsCollection(num, positions=pts)
    if isinstance(numbers, tuple):
        atoms.numbers[:] = random.choices(numbers, k=num)
    else:
        atoms.numbers[:] = numbers

    return atoms


class SeparationForce(functions.Function):
    """A separation force between atoms that come within the cutoff.  This can be used as a penalty in a loss function
    to"""

    supports_jacobian = True
    output_type = float

    def __init__(self, epsilon: float = 1, sigma=1.0, cutoff=1.0, power: int = 12):
        super().__init__()
        self._epsilon = epsilon
        self._sigma = sigma
        self._cutoff = cutoff
        self._power = power

        if cutoff is None:
            self._c1 = 0
            self._c2 = 0
        else:
            self._c1 = (
                self._power * self._sigma**self._power / cutoff ** (self._power + 1)
            )
            self._c2 = -((self._sigma / cutoff) ** self._power) - self._c1 * cutoff

    def evaluate(
        self, atoms: AtomsCollection, *, get_jacobian=False
    ):  # pylint: disable=arguments-differ
        distances = pdist(atoms.positions)
        total_energy = np.sum(tuple(map(self.energy, distances)))

        if get_jacobian:
            n = atoms.num_atoms  # pylint: disable=invalid-name
            jacobian = np.zeros((1, len(atoms)))
            positions = atoms.positions
            # Indexing to extract the ij distance from the pdist vec, see:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
            idx = lambda i, j: n * i + j - ((i + 2) * (i + 1)) // 2  # noqa: E731
            for i in range(n):
                for j in range(i + 1, n):
                    dr = positions[j] - positions[i]  # pylint: disable=invalid-name
                    dist = distances[idx(i, j)]
                    force = self.force(dist) * dr / dist

                    jacobian[0, atoms.linear_pos_idx(i)] += force
                    jacobian[0, atoms.linear_pos_idx(j)] -= force

            # return 0., np.zeros(jacobian.shape)
            return total_energy, jacobian

        # return 0.
        return total_energy

    @property
    def cutoff(self) -> float:
        """The distance at which this force and energy goes to zero"""
        return self._cutoff

    def energy(self, r: float) -> float:  # pylint: disable=invalid-name
        """
        :param r: separation between atoms
        :return: the corresponding energy
        """
        if self._cutoff is not None and r > self.cutoff:
            return 0.0

        energy = (
            4
            * self._epsilon
            * ((self._sigma / r) ** self._power + self._c1 * r + self._c2)
        )
        return energy

    def force(self, r: float):  # pylint: disable=invalid-name
        """
        :param r: separation between atoms
        :return: the corresponding force magnitude
        """
        if self._cutoff is not None and r > self.cutoff:
            return 0.0

        force = (
            4
            * self._epsilon
            * (
                self._power * self._sigma**self._power / r ** (self._power + 1)
                - self._c1
            )
        )
        return force
