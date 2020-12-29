# -*- coding: utf-8 -*-
from rdkit import Chem

from . import atomic


def milad2rdkit(atoms: atomic.AtomsCollection) -> Chem.Mol:
    """Covnert a MILAD atoms collection to an rdkit molecule"""
    mol = Chem.RWMol()
    conf = Chem.Conformer(atoms.num_atoms)
    for idx in range(atoms.num_atoms):
        atom = Chem.Atom(int(atoms.numbers[idx]))
        mol.AddAtom(atom)
        conf.SetAtomPosition(idx, atoms.positions[idx])
    mol.AddConformer(conf)
    return mol
