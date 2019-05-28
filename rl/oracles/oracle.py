from abc import abstractmethod
from rl.policies import Policy
from rl.tools.oracles import Oracle
from rl.tools.utils.misc_utils import safe_assign


class rlOracle(Oracle):
    """
    It should be deepcopy save.
    """

    def __init__(self, policy):
        self.policy = safe_assign(policy, Policy)

    @property
    @abstractmethod
    def ro(self):
        """Return the effective ro that defines this oracle."""

    @abstractmethod
    def update(ro, *args, **kwargs):
        """The update method should take ro as the first argument."""

    def __deepcopy__(self, memo=None, exclude=None):
        if memo is None:
            memo = {}
        if exclude is None:
            exclude = []
        exclude.append('policy')
        new = Oracle.__deepcopy__(self, memo, exclude=exclude)
        new.policy = self.policy
        return new
