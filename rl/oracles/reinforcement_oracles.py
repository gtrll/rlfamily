import numpy as np

from rl.tools.oracles import tfLikelihoodRatioOracle
from rl.oracles.oracle import rlOracle
from rl.policies import tfPolicy
from rl.tools.normalizers import OnlineNormalizer
from rl.tools.utils.tf_utils import tfObject
from rl.tools.utils import logz


class tfPolicyGradient(tfLikelihoodRatioOracle, rlOracle):
    """ A wrapper of tfLikelihoodRatioOracle for computing policy gradient of the type
            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]
        where \pi' is specified in ae.
    """

    @tfObject.save_init_args()
    def __init__(self, policy, ae, nor, correlated=True, use_log_loss=False, normalize_weighting=False,
                 onestep_weighting=True, avg_type='avg'):
        assert isinstance(policy, tfPolicy)
        assert isinstance(nor, OnlineNormalizer)
        rlOracle.__init__(self, policy)  # set policy as an attribute
        # Initialize the tfLikelihoodRatioOracle
        # we do not need to modify deepcopy lists because no new stateful objects are added.
        ph_args = [policy.ph_obs, policy.ph_acs]
        tfLikelihoodRatioOracle.__init__(
            self, policy.ts_vars, policy.ts_logp,
            ph_args=ph_args, nor=nor, correlated=correlated,
            use_log_loss=use_log_loss, normalize_weighting=normalize_weighting)
        # Define attributes for computing function values and weighting
        self._ae = ae
        self._onestep_weighting = onestep_weighting
        assert avg_type in ['avg', 'sum']
        self._avg_type = avg_type
        self._ro = None

    @property
    def _post_deepcopy_list(self):
        return tfLikelihoodRatioOracle._post_deepcopy_list.fget(self) + ['_ro']

    @property
    def ro(self):
        return self._ro

    def update_ae(self, ro, to_log=False, log_prefix=''):
        self._ae.update(ro, to_log=to_log, log_prefix=log_prefix)

    def update(self, ro, update_nor=False, shift_adv=False, to_log=False, log_prefix=''):
        """
            Args:
                ro: RO object representing the new information
                update_nor: whether to update the  control variate of tfLikelihoodRatioOracle
                shift_adv: whether to force the adv values to be positive. if float, it specifies the
                    amount to shift.
        """

        self._ro = ro  # save the ref to rollouts

        # Compute adv.
        advs, vfns = self._ae.advs(ro)  # adv has its own ref_policy
        adv = np.concatenate(advs)
        if shift_adv:  # make adv non-negative
            assert self._use_log_loss
            if shift_adv is True:
                adv = adv - np.min(adv)
            else:
                adv = adv - np.mean(adv) + shift_adv
            self._nor.reset()  # defined in tfLikelihoodRatioOracle
            update_nor = False

        if not self._normalize_weighting:
            if self._avg_type == 'sum':  # rescale the problem if needed
                adv *= len(adv) / len(ro)

        # Update the loss function.
        if self._use_log_loss is True:
            #  - E_{ob} E_{ac ~ q | ob} [ w * log p(ac|ob) * adv(ob, ac) ]
            if self._onestep_weighting:  # consider importance weight
                w_or_logq = np.concatenate(self._ae.weights(ro, policy=self.policy))  # helper function
            else:
                w_or_logq = np.ones_like(adv)
        else:  # False or None
            #  - E_{ob} E_{ac ~ q | ob} [ p(ac|ob)/q(ac|ob) * adv(ob, ac) ]
            assert self._onestep_weighting
            w_or_logq = ro.lps

        if to_log:
            vfn = np.concatenate(vfns)
            logz.log_tabular('max_adv', np.amax(np.abs(adv)))
            logz.log_tabular('max_vfn', np.amax(np.abs(vfn)))

        # Update the tfLikelihoodRatioOracle.
        super().update(-adv, w_or_logq, [ro.obs, ro.acs], update_nor)  # loss is negative reward
