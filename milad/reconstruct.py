# -*- coding: utf-8 -*-
from typing import Tuple, Optional

import numpy as np
from scipy import optimize

from . import atomic
from . import fingerprinting
from . import functions

__all__ = ('Decoder',)


class Decoder:

    def __init__(self, fingerprinter: fingerprinting.Fingerprinter):
        super().__init__()
        self._fingerprinter = fingerprinter

    def decode(
        self,
        fingerprint: functions.StateLike,
        starting_configuration: atomic.AtomsCollection,
        xtol=1e-5,
        atoms_builder: atomic.AtomsCollectionBuilder = None,
    ) -> atomic.AtomsCollection:
        """
        :param fingerprint: the fingerprint to decode back into an atoms collection
        :param starting_configuration: the starting atoms configuration
        :return: a decoded atoms collection
        """
        atoms_builder = atoms_builder or atomic.AtomsCollectionBuilder(starting_configuration.num_atoms)
        # We're going to need a residuals function
        residuals = functions.Chain(atoms_builder, self._fingerprinter.process, functions.Residuals(fingerprint))

        previous_result: Optional[Tuple] = None

        def calc(state: functions.StateLike):
            global previous_result
            # Calculate residuals and Jacobian
            res, jac = residuals(state, jacobian=True)
            print(f'{np.abs(res).max()}')
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
        starting_vec = functions.get_bare_vector(preprocess(starting_configuration)).flatten()

        result = optimize.least_squares(
            calc,
            x0=starting_vec,
            jac=jac,
            bounds=self._get_bounds(atoms_builder),
            xtol=xtol,
        )

        if not result.success:
            raise RuntimeError(f'Could not decode the passed input: {result.message}')

        final_atoms = atoms_builder(result.x)
        return preprocess.inverse(final_atoms)

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
