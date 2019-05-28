import numpy as np
from rl.algorithms.algorithm import Algorithm
from rl.policies import Policy
from rl import oracles as Or
from rl.online_learners import rlPiccoloFisher
from rl.tools.online_learners import Piccolo, PiccoloOpt
from rl.tools.utils.misc_utils import timed, safe_assign
from rl.tools.utils import logz


class PredictiveRL(Algorithm):
    """
        Accelerated policy learning through predictions given by model_oracle.

        This framework includes update rules: model-free, model-based, piccolo, dyna.
    """

    def __init__(self, pcl, oracle, policy, model_oracle=None,
                 update_rule='model-free',
                 update_in_pred=False,  # if to update normalizer and ae in prediction
                 take_first_pred=False,  # take a prediction step before the first correction step
                 warm_start=True,  # if to use first-order piccolo to warm start the VI problem
                 shift_adv=False,  # make the adv to be positive
                 stop_std_grad=False,  # freeze std
                 ignore_samples=False,  # ignore all the information from samples
                 ):

        self._pcl = safe_assign(pcl, Piccolo, PiccoloOpt)
        # Saved in the correction step to be used in prediction step.
        self._or = safe_assign(oracle, Or.rlOracle)
        self._policy = safe_assign(policy, Policy)
        if model_oracle is not None:
            self._mor = safe_assign(model_oracle, Or.rlOracle)

        self._itr = 0
        self._ro = None  # rollouts of from the environment
        self._g = None

        # flags
        assert update_rule in ['piccolo', 'model-free', 'model-based', 'dyna']
        self._update_rule = update_rule
        self._w_pred = update_rule in ['piccolo', 'dyna', 'model-based']
        self._pre_w_adap = update_rule in ['dyna', 'model-based']
        self._w_corr = update_rule in ['model-free', 'piccolo', 'dyna']

        self._update_in_pred = update_in_pred
        self._take_first_pred = take_first_pred
        self._warm_start = warm_start
        self._shift_adv = shift_adv
        self._stop_std_grad = stop_std_grad
        self._ignore_samples = ignore_samples

    def pi(self, ob):
        return self._policy.pi(ob, stochastic=True)

    def pi_ro(self, ob):
        return self._policy.pi(ob, stochastic=True)

    def logp(self, obs, acs):
        return self._policy.logp(obs, acs)

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            # Implement necessary pretraining procedures here.
            if isinstance(self._or, Or.tfPolicyGradient):
                self._ro = gen_ro(self.pi, logp=self.logp)
                self._or.update_ae(self._ro)

            # take a prediction step first
            if self._take_first_pred and self._w_pred:
                self._prediction()

    def update(self, ro):

        self._ro = ro
        if not self._ignore_samples:
            # update input normalizer for whitening
            self._policy.prepare_for_update(self._ro.obs)

            # Correction Step (Model-free)
            self._correction()

        # end of round
        self._itr += 1

        # log
        logz.log_tabular('pcl_stepsize', self._pcl.stepsize)
        logz.log_tabular('std', np.mean(self._policy.std))
        if not self._ignore_samples:
            logz.log_tabular('true_grads_size', np.linalg.norm(self._g))
            logz.log_tabular('pred_grads_size', np.linalg.norm(self._pcl.g_hat))
            pred_error_size = np.linalg.norm(self._g - self._pcl.g_hat)
            ratio = pred_error_size / np.linalg.norm(self._g)
            logz.log_tabular('pred_error_size', pred_error_size)
            logz.log_tabular('pred_error_true_ratio', ratio)

        # Prediction Step (Model-based)
        if self._w_pred:
            self._prediction()

        # log
        logz.log_tabular('std_after', np.mean(self._policy.std))

    def _correction(self):
        # single first-order update
        with timed('Update oracle'):
            self._or.update(self._ro, update_nor=True)
            if callable(getattr(self._or, 'update_ae')):
                self._or.update_ae(self._ro, to_log=True)

        with timed('Compute policy gradient'):
            g = self._or.compute_grad()
            self._g = g

        if self._w_corr:
            if self._update_rule in ['dyna', 'model_free']:
                self._pcl.clear_g_hat()  # make sure hat_g is None
            with timed('Take piccolo correction step'):
                kwargs = {}
                if isinstance(self._pcl, rlPiccoloFisher):
                    kwargs['ro'] = self._ro
                self._pcl.update(g, 'correct', **kwargs)

    def _prediction(self):
        # (multi-step) update using model-information

        with timed('Update model oracle'):
            # flags
            shift_adv = self._shift_adv and isinstance(self._pcl, PiccoloOpt)
            # if to update pol_nor and ae in model update
            update_ae_and_nor = self._pre_w_adap or self._update_in_pred

            # mimic the oracle update
            kwargs = {'update_nor': True, 'to_log': True}
            if isinstance(self._mor, Or.SimulationOracle):
                kwargs['update_ae'] = update_ae_and_nor
                kwargs['update_pol_nor'] = update_ae_and_nor
            elif (isinstance(self._mor, Or.LazyOracle)
                  or isinstance(self._mor, Or.AggregatedOracle)
                  or isinstance(self._mor, Or.AdversarialOracle)):
                kwargs['shift_adv'] = shift_adv
            elif isinstance(self._mor, Or.DummyOracle):
                kwargs['g'] = self._g
            else:
                raise NotImplementedError('Model oracle update is not implemented.')
            self._mor.update(ro=self._ro, **kwargs)

        with timed('Compute model gradient'):
            g_hat = self._mor.compute_grad()

        with timed('Take piccolo prediction step'):
            kwargs = {}
            if isinstance(self._pcl, rlPiccoloFisher):
                kwargs['ro'] = self._mor.ro

            if isinstance(self._pcl, PiccoloOpt):
                # need to define the optimization problem
                kwargs['grad_hat'] = self._mor.compute_grad
                kwargs['loss_hat'] = self._mor.compute_loss
                kwargs['warm_start'] = self._warm_start
                kwargs['stop_std_grad'] = self._stop_std_grad
                if isinstance(self._mor, Or.SimulationOracle):
                    def callback():
                        with timed('Update model oracle (callback)'):
                            self._mor.update(update_nor=True,
                                             update_ae=update_ae_and_nor,
                                             update_pol_nor=update_ae_and_nor)

                            method = getattr(self._pcl, 'method', None)
                            if isinstance(method, rlPiccoloFisher):
                                method.assign(self._policy)  # sync the normalizer
                                method.ro = self._mor.ro
                            if isinstance(self._pcl, rlPiccoloFisher):
                                self._pcl._reg_swp.update(self._mor.ro.obs)

                    kwargs['callback'] = callback

            # adapt for 'dyna' and 'model-based'
            self._pcl.update(g_hat, 'predict', adapt=self._pre_w_adap, **kwargs)
