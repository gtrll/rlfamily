from abc import ABC, abstractmethod
import numpy as np


class MovingAverage(ABC):
    """Moving average of numpy array."""

    @abstractmethod
    def update(self, new_val):
        # Update the moving average.
        pass

    @abstractmethod
    def replace(self, new_val):
        # Replace the last new_val with this new_val.
        pass

    @property
    @abstractmethod
    def val(self):
        # The value of the current estimate / the moving average.
        pass


class ExpMvAvg(MovingAverage):
    """An estimator based on exponential moving average.

    The estimate after N calls is computed as
        val = (1-rate) \sum_{n=1}^N rate^{N-n} x_n / nor_N
    where nor_N is equal to (1-rate) \sum_{n=1}^N rate^{N-n}
    """

    def __init__(self, init_val, rate, init_nor=0.):
        self._val = init_val * init_nor if init_val is not None else 0.
        self._nor = init_nor
        self.rate = rate
        self._old_val = self._val

    def update(self, new_val):
        self._val = self.mvavg(self._val, new_val, self.rate)
        self._nor = self.mvavg(self._nor, 1, self.rate)
        self._old_val = new_val

    def replace(self, new_val):
        self._val = self._val + (1.0 - self.rate) * (new_val - self._old_val)
        self._old_val = new_val

    @property
    def val(self):
        return self._val / np.maximum(1e-8, self._nor)

    @staticmethod
    def mvavg(old, new, rate):
        return rate * old + (1.0 - rate) * new


class MomentMvAvg(MovingAverage):
    """ An estimator based on momentum, without normalization.

        The estimate after N calls is computed as
            val = \sum_{n=1}^N rate^{N-n} x_n
    """

    def __init__(self, init_val, rate):
        self._val = init_val if init_val is not None else 0.
        self.rate = rate
        self._old_val = self._val

    def update(self, new_val):
        self._val = self.mvavg(self._val, new_val, self.rate)
        self._old_val = new_val

    def replace(self, new_val):
        self._val = self._val + (new_val - self._old_val)
        self._old_val = new_val

    @property
    def val(self):
        return self._val

    @staticmethod
    def mvavg(old, new, rate):
        return rate * old + new


class PolMvAvg(MovingAverage):
    """ An estimator based on polynomially weighted moving average.

        The estimate after N calls is computed as
            val = \sum_{n=1}^N n^power x_n / nor_N
        where nor_N is equal to \sum_{n=1}^N n^power, and power is a parameter.
    """

    def __init__(self, init_val, power=0, init_nor=0.):
        self._val = init_val * init_nor if init_val is not None else 0.
        self._nor = init_nor
        self.power = power
        self._itr = 1
        self._old_val = self._val

    def update(self, new_val):
        self._val = self.mvavg(self._val, new_val, self.power)
        self._nor = self.mvavg(self._nor, 1, self.power)
        self._old_val = new_val

    def replace(self, new_val):
        self._val = self._val + ((self._itr - 1)**self.power) * (new_val - self._old_val)
        self._old_val = new_val

    def mvavg(self, old, new, power):
        return old + new * self._itr**power

    @property
    def val(self):
        return self._val / self._nor
