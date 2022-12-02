# -*- coding: utf-8 -*-
import rdkit.Chem
from rdkit import Chem

from . import atomic

# Have to disable this warning because RDKit doesn't properly export its types
# pylint: disable=no-member


def milad2rdkit(atoms: atomic.AtomsCollection) -> "rdkit.Chem.Mol":
    """Convert a MILAD atoms collection to an rdkit molecule"""
    mol = Chem.RWMol()
    conf = Chem.Conformer(atoms.num_atoms)
    for idx in range(atoms.num_atoms):
        atom = Chem.Atom(int(atoms.numbers[idx]))
        mol.AddAtom(atom)
        conf.SetAtomPosition(idx, atoms.positions[idx])
    mol.AddConformer(conf)
    return mol


def rdkit2milad(conformer: "rdkit.Chem.Conformer") -> atomic.AtomsCollection:
    """Get a MILAD AtomsCollection from an RDKit conformer"""
    mol: Chem.Mol = conformer.GetOwningMol()
    return atomic.AtomsCollection(
        conformer.GetNumAtoms(),
        conformer.GetPositions(),
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
    )
