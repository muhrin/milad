# -*- coding: utf-8 -*-
import numpy as np

try:
    import dscribe.descriptors
except ImportError:
    __all__ = tuple()
else:
    from milad.play import asetools
    from milad import atomic
    from . import interfaces

    __all__ = ("DScribeDescriptor",)

    class DScribeDescriptor(interfaces.Descriptor):
        def __init__(self, dscribe_descriptor: dscribe.descriptors.Descriptor):
            super().__init__()
            self._dscribe = dscribe_descriptor

        @property
        def fingerprint_len(self) -> int:
            return self._dscribe.get_number_of_features()

        @property
        def cutoff(self) -> float:
            try:
                return getattr(self._dscribe, "_rcut")
            except AttributeError:
                return float("inf")

        def evaluate(self, atoms: atomic.AtomsCollection, *, get_jacobian=False):
            ase_atoms = asetools.milad2ase(atoms)

            if get_jacobian:
                natoms = atoms.num_atoms
                try:
                    res = self._dscribe.derivatives(ase_atoms, positions=[0])
                except AttributeError:
                    raise RuntimeError("Calculation of derivatives not supported")
                else:
                    jac = np.zeros((self.fingerprint_len, natoms * 4))
                    jac[:, : 3 * natoms] = (
                        res[0].reshape(natoms * 3, self.fingerprint_len).T
                    )
                    return res[1][0], jac

            return self._dscribe.create(ase_atoms, positions=[0])[0]
