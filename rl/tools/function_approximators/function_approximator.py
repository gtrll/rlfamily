from abc import ABC, abstractmethod
from functools import wraps


def online_compatible(f):
    # A decorator to make f, designed for batch inputs and outputs, support both single and
    # batch predictions
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        assert len(args) > 0
        return f(self, *(arg[None] for arg in args), **kwargs)[0] if len(args[0].shape) == 1 else f(self, *args, **kwargs)
    return wrapper


class FunctionApproximator(ABC):
    """
        An abstract interface of function approximators.

        It should be deepcopy compatible.
    """

    def __init__(self, x_dim, y_dim, name='Func', seed=None):
        self.name = name
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.seed = seed

    @online_compatible
    @abstractmethod
    def predict(self, x):
        """ Predict the values at x """

    @abstractmethod
    def prepare_for_update(self, *args, **kwargs):
        """ Perform pre-processing such as normalizer update before the update
        of trainable parameters. """

    @property
    @abstractmethod
    def variable(self):
        """ Return a np.ndarray of the trainable variables """

    @variable.setter
    @abstractmethod
    def variable(self, val):
        """ Set the trainable variables as val, which is a np.ndarray in the
        same format as self.variable."""

    # utilities
    @abstractmethod
    def assign(self, other):
        """Set the states of self (e.g. policy parameters) as other. In
        general, this is different from deepcopying other; with assign, self
        can have different, e.g., update behaviors from other.  """

    @abstractmethod
    def save(self, path, *args, **kwargs):
        """ Save the instance in path"""

    @abstractmethod
    def restore(self, path, *args, **kwargs):
        """ restore the saved instance in path """

    @abstractmethod
    def copy(self, name=None):
        """ Like copy.deepcopy but with a new name. If name is None, use the
        old name"""
