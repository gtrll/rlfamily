from abc import ABC, abstractmethod


class Algorithm(ABC):

    @abstractmethod
    def pretrain(self, gen_ro):
        """ Pretrain the policy. """

    @abstractmethod
    def update(self, ro):
        """ Update the policy based on rollouts. """

    @abstractmethod
    def pi(self, ob):
        """ Target policy for online querying. """

    @abstractmethod
    def pi_ro(self, ob):
        """ Behavior policy for online querying. """

    @abstractmethod
    def logp(self, obs, acs):
        """ Log probability for batch querying. """
