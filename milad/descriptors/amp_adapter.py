# -*- coding: utf-8 -*-
try:
    import amp  # pylint: disable=unused-import
except ImportError:
    __all__ = tuple()
else:
    from amp import utilities
    from ase.calculators.calculator import Parameters

    from milad.play import asetools

    __all__ = ('AmpDescriptor',)

    class AmpDescriptor:
        """This adapter allows MILAD descriptors to be used by the amp code:
         https://amp.readthedocs.io/en/latest/

         Functionality is not complete but the basics do work.
         """

        def __init__(self, descriptor, dblabel=None):
            self._descriptor = descriptor
            self.dblabel = dblabel
            self.parent = None
            self.fingerprints = None
            self.parameters = Parameters({
                'importname': 'milad.descriptors.amp_adapter.AmpDescriptor',
                'mode': 'atom-centered'
            })

        def tostring(self) -> str:
            """Returns an evaluatable representation of the calculator that can
            be used to restart the calculator."""
            return self.parameters.tostring()

        def calculate_fingerprints(
            # pylint: disable=unused-argument
            self,
            images,
            parallel=None,
            log=None,
            calculate_derivatives=False
        ):
            if self.dblabel is None:
                if hasattr(self.parent, 'dblabel'):
                    self.dblabel = self.parent.dblabel
                else:
                    self.dblabel = 'amp-data'

            if self.fingerprints is None:
                self.fingerprints = utilities.Data(filename=f'{self.dblabel}-fingerprints')

            self.fingerprints.open()
            for hashval, atoms in images.items():
                fps = []
                for env in asetools.extract_environments(atoms, cutoff=self._descriptor.cutoff):
                    milad_env = asetools.ase2milad(env)
                    fps.append((env.symbols[0], self._descriptor(milad_env)))
                self.fingerprints.d[hashval] = fps
            self.fingerprints.close()
