from abc import abstractmethod
import numpy as np
import copy
from rl.tools.utils.mvavg import ExpMvAvg
from rl.tools.oracles.oracle import Oracle


class MetaOracle(Oracle):
    """These Oracles are built on other Oracle objects."""

    @abstractmethod
    def __init__(self, base_oracle, *args, **kwargs):
        """It should have attribute base_oracle or base_oracles."""


class DummyOracle(MetaOracle):

    def __init__(self, base_oracle, *args, **kwargs):
        self._base_oracle = copy.deepcopy(base_oracle)
        self._g = 0.

    def compute_loss(self):
        return 0.

    def compute_grad(self):
        return self._g

    def update(self, g=None, *args, **kwargs):
        assert g is not None
        self._base_oracle.update(*args, **kwargs)
        self._g = np.copy(g)


class LazyOracle(MetaOracle):
    """Function-based oracle based on moving average."""

    def __init__(self, base_oracle, beta):
        self._base_oracle = copy.deepcopy(base_oracle)
        self._beta = beta
        self._f = ExpMvAvg(None, beta)
        self._g = ExpMvAvg(None, beta)

    def update(self, *args, **kwargs):
        self._base_oracle.update(*args, **kwargs)
        self._f.update(self._base_oracle.compute_loss())
        self._g.update(self._base_oracle.compute_grad())

    def compute_loss(self):
        f = self._base_oracle.compute_loss()
        return f

    def compute_grad(self):
        g = self._base_oracle.compute_grad()
        return g


class AdversarialOracle(LazyOracle):
    """For debugging purpose."""

    def __init__(self, base_oracle, beta):
        super().__init__(base_oracle, beta)
        self._max = None

    def compute_grad(self):
        g = super().compute_grad()
        if self._max is None:
            self._max = np.linalg.norm(g)
        else:
            self._max = max(self._max, np.linalg.norm(g))
        return -g / max(np.linalg.norm(g), 1e-5) * self._max
