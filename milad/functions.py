# -*- coding: utf-8 -*-
import abc
import logging
from typing import List, Union, Tuple, Callable, Any, Iterable, Type, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


class State(metaclass=abc.ABCMeta):
    """A state vector representing an input to a function"""

    @property
    def dtype(self):
        return self.vector.dtype

    @property
    @abc.abstractmethod
    def vector(self) -> np.array:
        """Get the state vector as a numpy array"""

    @property
    def array(self) -> np.array:
        """Get the state as a numpy array"""
        return self.vector

    def __len__(self) -> int:
        """Get the length of this state vector"""
        return len(self.vector)

    @property
    def length(self) -> int:
        """Get the length of this state vector"""
        return len(self.vector)


StateLike = Union[np.ndarray, State]


class PlainState(State):
    __slots__ = ('_array',)

    def __init__(self, length):
        self._array = np.zeros(length)

    @property
    def vector(self) -> np.array:
        return self._array

    @property
    def array(self) -> np.array:
        """Get the array corresponding to this state"""
        return self._array

    @array.setter
    def array(self, arr: np.ndarray):
        self._array[:] = arr


class Feature(State):
    LENGTH = None

    def __init__(self, dtype=float):
        super().__init__()
        if self.LENGTH is None:
            raise RuntimeError('Feature length not set, please set the LENGTH class attribute')
        self._vector = np.zeros(self.LENGTH, dtype=dtype)

    @property
    def vector(self):
        return self._vector

    @property
    def array(self):
        return self._vector

    @array.setter
    def array(self, arr: np.ndarray):
        self._vector[:] = arr

    @vector.setter
    def vector(self, value: np.array):
        if not value.size == self._vector.size:
            raise ValueError(f"Size mismatch, expected '{self._vector.size}', got '{value.size}'")
        self._vector[:] = value[:]

    def __add__(self, other: 'Feature') -> 'Features':
        if not isinstance(other, Feature):
            raise TypeError(f"Don't know how to add '{other.__class__.__name__}' to a feature")
        return Features(self, other)


class Features(State):
    """An collection of features"""

    def __init__(self, *feature: Feature):
        self._vector = np.empty(0)
        self._features: List[Feature] = []
        for entry in feature:
            self.add(entry)

    @property
    def features(self) -> List[Feature]:
        """Access the list of features"""
        return self._features

    @property
    def vector(self):
        return self._vector

    @property
    def array(self):
        return self._vector

    @array.setter
    def array(self, arr):
        self._vector[:] = arr

    def add(self, feature: Feature):
        self._features.append(feature)
        self._vector = np.append(self._vector, feature.vector)
        # Reassign all the arrays
        idx = 0
        for entry in self._features:
            entry.array = self._vector[idx:idx + len(entry)]
            idx += entry.length

    def __add__(self, other: Union[Feature, 'Features']) -> 'Features':
        if not isinstance(other, (Feature, Features)):
            raise TypeError(f"Don't know how to add a '{other.__class__.__name__}' to an environment")
        return Features(*self.features, other)


class WeightedDelta(Feature):
    """A Dirac delta function with a position and a weight (the total integral) stored as
    [x, y, z, weight]
    """
    # Indexing helpers
    X = 0
    Y = 1
    Z = 2
    WEIGHT = 3
    LENGTH = 4

    def __init__(self, pos: np.array, weight=1.0):
        super().__init__(dtype=pos.dtype)
        self.pos = pos
        self.weight = weight

    def __repr__(self) -> str:
        return f'Delta(pos={self.pos}, weight={self.weight})'

    @property
    def pos(self) -> np.array:
        return self.vector[:3]

    @pos.setter
    def pos(self, value):
        self.vector[:3] = value

    @property
    def weight(self):
        return self.vector[3]

    @weight.setter
    def weight(self, value):
        self.vector[3] = value


class WeightedGaussian(Feature):
    """A 3 dimensional gaussian stored as [x, y, z, sigma, weight]"""
    # Indexing helpers
    X = 0
    Y = 1
    Z = 2
    SIGMA = 3
    WEIGHT = 4
    LENGTH = 5

    def __init__(self, pos: np.array, sigma: float = 1., weight: float = 1.):
        super().__init__()
        self.pos = pos
        self.sigma = sigma
        self.weight = weight

    def __repr__(self) -> str:
        return f'Gaussian(pos={self.pos}, sigma={self.sigma}, weight={self.weight})'

    @property
    def pos(self) -> np.array:
        return self.vector[:3]

    @pos.setter
    def pos(self, value):
        self.vector[:3] = value

    @property
    def sigma(self) -> float:
        return self.vector[self.SIGMA]

    @sigma.setter
    def sigma(self, value):
        self.vector[self.SIGMA] = value

    @property
    def weight(self) -> float:
        return self.vector[self.WEIGHT]

    @weight.setter
    def weight(self, value):
        self.vector[self.WEIGHT] = value


# region Functions


class Function(metaclass=abc.ABCMeta):
    input_type = (State, np.ndarray)
    output_type = (State, np.ndarray)
    supports_jacobian = False

    def __init__(self):
        self._callbacks = set()

    @property
    def inverse(self) -> Optional['Function']:
        """A function may optionally provide an inverse in which case it would be returned by this property"""
        return None

    def add_callback(self, func: Callable):
        if func in self._callbacks:
            raise ValueError("'{}' is already registered".format(func))
        self._callbacks.add(func)

    def remove_callback(self, func: Callable[[StateLike, StateLike, np.ndarray], None]):
        self._callbacks.remove(func)

    def __call__(self, state: State, jacobian=False):
        name = self.__class__.__name__

        if self.input_type is not None:
            self._check_input_type(state, self.input_type)

        result = self.evaluate(state, get_jacobian=jacobian)
        if result is None:
            raise RuntimeError(f'{name} produced None output')

        if jacobian:
            if not isinstance(result, tuple):
                raise RuntimeError(f"{name}.evaulate didn't return Jacobian despite being asked to")

            _LOGGER.debug(
                '%s: |input|: %s, |output|: %s, |jac|: %s', name, np.mean(get_bare_vector(state)),
                np.mean(get_bare_vector(result[0])), np.mean(result[1])
            )
        else:
            if np.isnan(get_bare_vector(result)).any():
                raise ValueError(f'{name}.evaulate produce a result with a NaN entry')
            _LOGGER.debug(
                '%s: |input|: %s, |output|: %s', name, np.mean(get_bare_vector(state)),
                np.mean(get_bare_vector(result))
            )

        for func in self._callbacks:
            if jacobian:
                func(state, result[0], result[1])
            else:
                func(state, result, jacobian=None)
        return result

    @abc.abstractmethod
    def evaluate(self, state, get_jacobian=False):
        """Evaluate the function with the passed input

        :param state: the state that is the input to the function
        :param get_jacobian: if True returns a tuple [value, jacobian_mtx] else returns value
        """

    @classmethod
    def _check_input_type(cls, value: Any, allowed_types: Union[Type, Iterable[Type]]):
        cls._check_type(value, allowed_types, 'Unsupported input type')

    @classmethod
    def _check_output_type(cls, value: Any, allowed_types: Union[Type, Iterable[Type]]):
        cls._check_type(value, allowed_types, 'Unsupported output type')

    @classmethod
    def _check_type(cls, value: Any, allowed_types: Union[Type, Iterable[Type]], msg: str):
        if not isinstance(value, allowed_types):
            raise TypeError(
                f'{cls.__name__}: {msg}, '
                f"expected '{allowed_types}', "
                f"got '{value.__class__.__name__}'"
            )


class Identity(Function):
    """Function that just returns what it is passed"""

    def evaluate(self, state: StateLike, get_jacobian=False):
        """Evaluate the function with the passed input"""
        if get_jacobian:
            return state, np.eye(len(state))

        return state


class Chain(Function):
    """A function that is a chain of other functions."""

    def __init__(self, *functions):
        super().__init__()
        self._functions: List[Function] = list(functions)

    def __len__(self):
        return len(self._functions)

    def __getitem__(self, item) -> Function:
        if isinstance(item, slice):
            return Chain(*self._functions[item])

        return self._functions[item]

    def copy(self) -> 'Chain':
        """Create a shallow copy"""
        return Chain(self._functions)

    def find_type(self, query: type) -> List[Tuple[int, Function]]:
        """Look for functions in this chain of the given type.  Returns a list of Tuple[index, function]
        where index is the index in the chain and function is the matching function instance
        """
        found = []
        for idx, func in enumerate(self):
            if isinstance(func, query):
                found.append((idx, func))
        return found

    @property
    def input_type(self):
        if not self._functions:
            return None

        return self._functions[0].input_type

    @property
    def output_type(self):
        if not self._functions:
            return None

        return self._functions[-1].output_type

    @property
    def supports_jacobian(self):
        return all(entry.supports_jacobian for entry in self._functions)

    @property
    def inverse(self) -> Optional['Function']:
        inverse_chain = Chain()
        for func in reversed(self._functions):
            inverse = func.inverse
            if inverse is None:
                # Can't build inverse chain as at least one function doesn't have an inverse
                return None
            inverse_chain.append(inverse)

        return inverse_chain

    def append(self, function):
        self._functions.append(function)

    def evaluate(self, state: StateLike, get_jacobian=False):
        if len(self._functions) == 1:
            # Just pipe directly through
            return self._functions[-1](state, get_jacobian)

        current_in = state
        previous_jacobian = None

        # Do all but last
        for function in self._functions[:-1]:
            # Compute values for the current function
            current_out = function(current_in, get_jacobian)
            if get_jacobian:
                current_out, current_jacobian = current_out

                if previous_jacobian is None:
                    # This is the first time around, so no previous to multiply with
                    previous_jacobian = current_jacobian
                else:
                    # Apply chain rule and propagate jacobians
                    previous_jacobian = np.matmul(current_jacobian, previous_jacobian)

            # Make this output the input to the next function
            current_in = current_out

        # Do the last one
        function = self._functions[-1]

        current_out = function(current_in, get_jacobian)
        if get_jacobian:
            current_out, current_jacobian = current_out
            return current_out, np.matmul(current_jacobian, previous_jacobian)

        return current_out


class Residuals(Function):
    """Given some measurements (m) calculate the residuals from some data (d) as f = d - m """
    supports_jacobian = True

    def __init__(self, data: np.array):
        super().__init__()
        self._data = get_bare_vector(data)

    def output_length(self, in_state: State) -> int:
        return len(self._data)

    def empty_jacobian(self, in_state: State) -> np.array:
        return np.empty((self.output_length(in_state), len(in_state)), dtype=complex)

    def evaluate(self, state: State, get_jacobian=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        vector = get_bare_vector(state)
        out_vector = vector - self._data
        if get_jacobian:
            return out_vector, np.identity(len(state), dtype=out_vector.dtype)

        return out_vector


class Native(Function):

    def __init__(self, func: Callable[[StateLike], StateLike], jac: Callable[[StateLike], np.ndarray]):
        super().__init__()
        self._fn = func
        self._jac = jac

    @property
    def support_jacobian(self):
        return self._jac is not None

    def evaluate(self, state: State, get_jacobian=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.supports_jacobian:
            return self._fn(state), self._jac(state)

        return self._fn(state)


# endregion


def get_bare_vector(state: Union[np.ndarray, State]) -> np.array:
    if isinstance(state, np.ndarray):
        return state
    if isinstance(state, State):
        return state.vector

    raise TypeError(f"Unknown state type: '{state.__class__.__name__}'")
