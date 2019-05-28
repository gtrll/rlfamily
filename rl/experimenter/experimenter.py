import functools
import time
import numpy as np
from rl.algorithms import Algorithm
from rl.experimenter.generate_rollouts import generate_rollout
from rl.tools.utils.misc_utils import safe_assign, timed
from rl.tools.utils import logz


class Experimenter:

    def __init__(self, alg, env, ro_kwargs):
        """
        ro_kwargs is a dict with keys, 'min_n_samples', 'max_n_rollouts', 'max_rollout_len'
        """
        self._alg = safe_assign(alg, Algorithm)
        self._gen_ro = functools.partial(generate_rollout, env=env, **ro_kwargs)
        self._ndata = 0  # number of data points seen

    def gen_ro(self, pi, logp=None, log_prefix='', to_log=False):
        ro = self._gen_ro(pi, logp)
        self._ndata += ro.n_samples
        if to_log:
            log_rollout_info(ro, prefix=log_prefix)
            logz.log_tabular(log_prefix + 'NumberOfDataPoints', self._ndata)

        return ro

    def run_alg(self, n_itrs, pretrain=True, save_policy=False, save_freq=100, final_eval=False):
        start_time = time.time()
        if pretrain:  # algorithm-specific
            self._alg.pretrain(functools.partial(self.gen_ro, to_log=False))

        # Main loop
        for itr in range(n_itrs):
            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", itr)
            with timed('Generate env rollouts'):
                ro = self.gen_ro(self._alg.pi_ro, logp=self._alg.logp, to_log=True)
            self._alg.update(ro)  # algorithm-specific
            logz.dump_tabular()  # dump log


def log_rollout_info(ro, prefix=''):
    # print('Logging rollout info')
    if not hasattr(log_rollout_info, "total_n_samples"):
        log_rollout_info.total_n_samples = {}  # static variable
    if prefix not in log_rollout_info.total_n_samples:
        log_rollout_info.total_n_samples[prefix] = 0
    sum_of_rewards = [rollout.rws.sum() for rollout in ro.rollouts]
    rollout_lens = [len(rollout) for rollout in ro.rollouts]
    n_samples = sum(rollout_lens)
    log_rollout_info.total_n_samples[prefix] += n_samples
    logz.log_tabular(prefix + "NumSamplesThisBatch", n_samples)
    logz.log_tabular(prefix + "NumberOfRollouts", len(ro))
    logz.log_tabular(prefix + "TotalNumSamples", log_rollout_info.total_n_samples[prefix])
    logz.log_tabular(prefix + "MeanSumOfRewards", np.mean(sum_of_rewards))
    logz.log_tabular(prefix + "StdSumOfRewards", np.std(sum_of_rewards))
    logz.log_tabular(prefix + "MaxSumOfRewards", np.max(sum_of_rewards))
    logz.log_tabular(prefix + "MinSumOfRewards", np.min(sum_of_rewards))
    logz.log_tabular(prefix + "MeanRolloutLens", np.mean(rollout_lens))
    logz.log_tabular(prefix + "StdRolloutLens", np.std(rollout_lens))
    logz.log_tabular(prefix + "MeanOfRewards", np.sum(sum_of_rewards) / (n_samples + len(sum_of_rewards)))
