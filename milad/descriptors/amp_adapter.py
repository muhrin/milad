# -*- coding: utf-8 -*-

try:
    import amp  # pylint: disable=unused-import
except ImportError:
    __all__ = tuple()
else:
    import argparse
    import collections
    from typing import Dict, Any

    from amp import utilities
    import ase
    from ase.calculators.calculator import Parameters
    import numpy as np

    from milad.play import asetools
    from . import interfaces

    __all__ = ('AmpDescriptor',)

    class AmpDescriptor:
        """This adapter allows MILAD descriptors to be used by the amp code:
         https://amp.readthedocs.io/en/latest/

         Functionality is not complete but the basics do work.
         """

        def __init__(self, descriptor: interfaces.Descriptor, dblabel=None):
            self._descriptor = descriptor
            self.dblabel = dblabel
            self.parent = None
            self.parameters = Parameters({'cutoff': descriptor.cutoff, 'mode': 'atom-centered'})

        @property
        def cutoff(self) -> float:
            return self._descriptor.cutoff

        def load_amp(self, filename: str, label='') -> amp.Amp:
            """Given a checkpoint filename this will load the Amp class and pass it this descriptor.

            This means that the caller needs to be sure that this descriptor is the one used for when training the
            loaded model, otherwise there will be inconsistencies.
            """
            if not label:
                label = filename.replace('.amp', '')
            return amp.Amp.load(filename, Descriptor=lambda *_args, **kwargs: self, label=label)

        def tostring(self) -> str:
            """Returns an evaluatable representation of the calculator that can
            be used to restart the calculator."""
            return self.parameters.tostring()

        def calculate_fingerprints(
            # pylint: disable=unused-argument
            self,
            images: Dict[Any, ase.Atoms],
            parallel=None,
            log=None,
            calculate_derivatives=False
        ):
            log = utilities.Logger(file=None) if log is None else log

            if self.dblabel is None:
                if hasattr(self.parent, 'dblabel'):
                    self.dblabel = self.parent.dblabel
                else:
                    self.dblabel = 'amp-data'

            log('Fingerprinting images...', tic='fp')
            if not hasattr(self, 'fingerprints'):
                # pylint: disable=attribute-defined-outside-init
                self.fingerprints = utilities.Data(
                    filename=f'{self.dblabel}-fingerprints',
                    calculator=argparse.Namespace(calculate=self._calc_fingerprint)
                )

            self.fingerprints.calculate_items(images, parallel=dict(cores=1), log=log)
            log('...fingerprints calculated.', toc='fp')

            if calculate_derivatives:
                log('Calculating fingerprint derivatives of images...', tic='derfp')
                if not hasattr(self, 'fingerprintprimes'):
                    # pylint: disable=attribute-defined-outside-init
                    self.fingerprintprimes = utilities.Data(
                        filename=f'{self.dblabel}-fingerprints-primes',
                        calculator=argparse.Namespace(calculate=self._calc_fingerprint_derivatives)
                    )  # pylint: disable=attribute-defined-outside-init

                self.fingerprintprimes.calculate_items(images, parallel=dict(cores=1), log=log)
                log('...fingerprint derivatives calculated.', toc='derfp')

        def _calc_fingerprint(self, system: ase.Atoms, _hashval):
            fingerprints = []

            for my_idx, env in asetools.extract_environments(
                system, cutoff=self._descriptor.cutoff, yield_indices=True, include_central_atom=False
            ):
                my_symbol = system.symbols[my_idx]
                milad_env = asetools.ase2milad(env)
                fingerprints.append((my_symbol, self._descriptor(milad_env)))

            return fingerprints

        def _calc_fingerprint_derivatives(self, system: ase.Atoms, _hashval):
            # pylint: disable=too-many-locals
            fp_length = self._descriptor.fingerprint_len
            derivatives = collections.defaultdict(lambda: np.zeros(fp_length))

            for my_idx, env in asetools.extract_environments(
                system,
                cutoff=self._descriptor.cutoff,
                yield_indices=True,
                include_central_atom=False,
            ):
                my_symbol = system.symbols[my_idx]
                milad_env = asetools.ase2milad(env)

                _, jac = self._descriptor(milad_env, jacobian=True)

                natoms = len(env)  # Number of atoms in the environment

                # The yielded environment has this array that allows us to map back on to the index in the
                # original structure
                orig_indices = env.get_array('orig_indices', copy=False)
                derivs = collections.defaultdict(lambda: np.zeros((3, fp_length)))

                for i in range(natoms):
                    neighbour_idx = orig_indices[i]
                    if neighbour_idx == my_idx:
                        continue

                    # Add to the derivatives as the same neighbour may contribute more than once
                    local_derivs = jac[:, i * 3:(i + 1) * 3].T
                    derivs[neighbour_idx] += local_derivs

                # Now copy over the derivatives to AMP format
                for neighbour_idx, derivs_ in derivs.items():
                    neighbour_symbol = system.symbols[neighbour_idx]

                    for coord in range(3):
                        # Force on this atom because of neighbour within this environment
                        deriv_idx = (my_idx, my_symbol, my_idx, my_symbol, coord)
                        derivatives[deriv_idx] += -derivs_[coord]

                        # Forces on neighbours because of this atom
                        deriv_idx = (neighbour_idx, neighbour_symbol, my_idx, my_symbol, coord)
                        derivatives[deriv_idx] = derivs_[coord]

            return {key: val.real.tolist() for key, val in derivatives.items()}
