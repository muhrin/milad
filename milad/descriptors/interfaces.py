# -*- coding: utf-8 -*-
import abc

from milad import atomic
from milad import functions

__all__ = ("Descriptor",)


class Descriptor(functions.Function):
    input_type = atomic.AtomsCollection

    @property
    @abc.abstractmethod
    def fingerprint_len(self) -> int:
        """Get the length of the fingerprint vector that will be returned"""

    @property
    @abc.abstractmethod
    def cutoff(self) -> float:
        """Get the descriptor cutoff radius"""

    @abc.abstractmethod
    def evaluate(
        self, atoms: atomic.AtomsCollection, *, get_jacobian=False
    ):  # pylint: disable=arguments-differ
        """Custom evaluate for descriptors.  These always take an AtomsCollection"""
