import numpy as np
import copy


class RO(object):
    """ A collection of Rollout instances.

        It has attributes:
            rollouts: a list of Rollout instances, each records one rollout.
            obs, acs, rws, sts, lps: np.ndarrays of concatenated statistics
            max_n_rollouts: the maximal number of rollouts to keep.
    """

    def __init__(self, rollouts, max_n_rollouts=None, max_n_samples=None):
        """rollouts: a list of Rollouts."""
        assert type(rollouts) is list and type(rollouts[0]) is Rollout
        self.rollouts = copy.deepcopy(rollouts)
        self.obs, self.acs, self.rws, self.sts, self.lps = self.flatten('all')
        if max_n_rollouts is not None:
            max_n_rollouts = max(max_n_rollouts, len(self.rollouts))
        self.max_n_rollouts = max_n_rollouts
        self.max_n_samples = max_n_samples

    def append(self, rollouts):
        """rollouts: a list of Rollouts."""
        assert type(rollouts) is list and type(rollouts[0]) is Rollout
        rollouts = copy.deepcopy(rollouts)
        # check if we need to throw away some old rollouts
        if self.max_n_rollouts is not None:
            extra_n_rollouts = len(self.rollouts) + len(rollouts) - self.max_n_rollouts
            if extra_n_rollouts > 0:  # throw away some old rollouts
                n_rollouts = len(self.rollouts) - extra_n_rollouts  # keep only most recent n_rollouts
                self.rollouts = [] if n_rollouts <= 0 else self.rollouts[-n_rollouts:]

        # now we're safe to append (always keep all the new rollouts)
        self.rollouts += copy.deepcopy(rollouts)

        # check if we still need throw away some samples
        if self.max_n_samples is not None:
            ns = np.array([len(rollout) for rollout in self.rollouts])
            n_samples = np.sum(ns)
            extra_n_samples = n_samples - self.max_n_samples
            if extra_n_samples > 0:
                ind = np.argwhere(np.flipud(ns).cumsum() > self.max_n_samples)
                ind = ind[1][0] if len(ind) > 1 else ind[0][0]  # let's allow having more than less
                self.rollouts = self.rollouts[-ind:]

        # update the continuous copy
        self.obs, self.acs, self.rws, self.sts, self.lps = self.flatten('all')
        print('n_samples', self.n_samples, 'len', len(self))

    def flatten(self, field):
        # flatten list of Rollouts into a single array
        if field == 'all':
            obs = self.flatten('obs_short')
            acs = self.flatten('acs')
            rws = self.flatten('rws')
            sts = self.flatten('sts_short')
            lps = self.flatten('lps')
            assert len(obs) == len(sts) == len(rws) == len(acs) == len(lps)
            return obs, acs, rws, sts, lps
        else:
            return np.concatenate([getattr(rollout, field) for rollout in self.rollouts])

    def __add__(self, other):
        assert type(other) is RO
        rollouts = self.rollouts + other.rollouts

        def sum(a, b):
            return None if (a is None or b is None) else a+b

        max_n_rollouts = sum(self.max_n_rollouts, other.max_n_rollouts)
        max_n_samples = sum(self.max_n_samples, other.max_n_samples)

        return RO(rollouts, max_n_rollouts=max_n_rollouts,
                  max_n_samples=max_n_samples)

    @property
    def n_samples(self):
        return len(self.obs)

    def __len__(self):
        return len(self.rollouts)


class Rollout(object):

    def __init__(self, obs, acs, rws, sts, done, logp=None):
        # obs, acs, rws, sts, lps are lists of vals
        assert len(obs) == len(sts)
        assert len(acs) == len(rws)
        assert len(obs) == len(acs) + 1
        self.obs = np.array(obs)
        self.acs = np.array(acs)
        self.rws = np.array(rws)
        self.sts = np.array(sts)
        self.done = done
        if logp is None:  # viewed as deterministic
            self.lps = np.zeros(self.acs.shape)
        else:
            self.lps = logp(self.obs_short, self.acs)

    @property
    def obs_short(self):
        return self.obs[:-1]

    @property
    def sts_short(self):
        return self.sts[:-1]

    def __len__(self):
        return len(self.rws)
