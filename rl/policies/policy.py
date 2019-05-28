from abc import abstractmethod
from rl.tools.function_approximators import FunctionApproximator, online_compatible


class Policy(FunctionApproximator):  # a policy is namely a stochastic FunctionApproximator
    """
    An abstract interface that represents conditional distribution \pi(a|s).

    It should be deepcopy compatible.
    """

    def __init__(self, x_dim, y_dim, name='Policy', seed=None):
        super().__init__(x_dim, y_dim, name=name, seed=seed)

    @property
    def ob_dim(self):
        return self.x_dim

    @property
    def ac_dim(self):
        return self.y_dim

    @online_compatible
    @abstractmethod
    def pi(self, obs, stochastic=True):
        """ Return the actions at obs; the policy, which can be stochastic"""

    @online_compatible
    @abstractmethod
    def logp(self, obs, acs):
        """ Return the log probability (on a batch of obs and acs) as a np.ndarray """

    @abstractmethod
    def kl(self, other, obs, reversesd=False):
        """ Computes KL(self||other), where other is another object of the
        # same policy class. If reversed is True, return KL(other||self) """

    @abstractmethod
    def fvp(self, obs, g):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable """
