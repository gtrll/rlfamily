from abc import ABC, abstractmethod
import copy


class Oracle(ABC):
    """
        An Oracle defines the objective function for an optimization problem.

        'compute_loss' and 'compute_grad' provide the interface for the
        optimization routine. 'update' redefines the objective function.

        A child class needs to support __deepcopy__.
    """
    @abstractmethod
    def compute_loss(self):
        # return a scalar
        pass

    @abstractmethod
    def compute_grad(self):
        # return an np.ndarray
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def __deepcopy__(self, memo=None, exclude=None):
        """ __deepcopy__ but with an exclusion list
            exclude is a list of attribute names (string) that is to be shallow copied.
        """
        if memo is None:
            memo = {}
        if exclude is None:
            exclude = []

        new = copy.copy(self)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k not in exclude:
                setattr(new, k, copy.deepcopy(v, memo))
        return new
