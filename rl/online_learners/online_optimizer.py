import numpy as np
import tensorflow as tf
import rl.tools.utils.tf_utils as U
from rl.tools.online_learners import online_optimizer as OO
from rl.tools.online_learners import base_algorithms as BA
from rl.policies import Policy


class rlOnlineOptimizer(object):
    pass


def _rlOnlineOptimizerDecorator(cls):
    """
    A decorator for reusing OO.OnlineOptimizers while making them use policy as the variable.
    This would create a new class that is both the base class in OO.OnlineOptimizer and rlOnlineOptimizer.
    """
    assert issubclass(cls, OO.OnlineOptimizer)
    class decorated_cls(cls, rlOnlineOptimizer):
        """ self._policy is only used as an interface variable for other rl
        codes, which can be modified on-the-fly, NOT the state of the optimzier
        """

        def __init__(self, policy, base_alg, p=0.0, **kwargs):
            assert isinstance(policy, Policy) and isinstance(base_alg, BA.BaseAlgorithm)
            super().__init__(base_alg, p=p, **kwargs)
            self._policy = policy

        def update(self, *args, **kwargs):
            super().update(*args, **kwargs)
            self._policy.variable = self.x

        def _set_x(self, val):
            super()._set_x(val)
            self._policy.variable=self.x

        def assign(self, policy):
            # NOTE this does not change the state of the optimizer
            self._policy.assign(policy)

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


class rlPiccoloFisher(object):
    pass


def _rlPiccoloFisherDecorator(cls):
    """
    A class decorator for using Fisher information matrix (and KL divergence)
    as the regularization in a Piccolo class that has been decorated as an
    rlOnlineOptimizer.
    """
    assert issubclass(cls, OO.Piccolo)
    if not issubclass(cls, rlOnlineOptimizer):
        cls = _rlOnlineOptimizerDecorator(cls)

    class Reg(object):
        # regularization based on KL divergence between policies
        def __init__(self, refpol, varpol, default_damping=0.1):
            """
            refpol:
                reference point to compute gradient.
            varpol:
                variable policy, which has the variables to optimize over.
            """
            assert isinstance(refpol, Policy) and isinstance(varpol, Policy)
            self._damping0 = default_damping
            self.refpol = refpol
            self.varpol = varpol
            self.obs = None
            self.damping = None

        def kl(self, x):
            self.varpol.variable = x
            return self.refpol.kl(self.varpol, self.obs, reversesd=False)

        def fvp(self, g):
            return self.refpol.fvp(self.obs, g) + self.damping * g

        @property
        def std(self):
            return self.refpol.std if hasattr(self.refpol, 'std') else 1.0

        def assign(self, reg):
            assert type(self) == type(reg)
            self.refpol.assign(reg.refpol)
            self.obs = np.copy(reg.obs)
            self.damping = np.copy(reg.damping)
            self._damping0 = np.copy(reg._damping0)

        def update(self, obs):
            self.refpol.assign(self.varpol)
            self.obs = np.copy(obs)
            self.damping = self._damping0  # /np.mean(self.std**2.0)

        @property
        def initialized(self):
            return not self.obs is None

    # We use Fisher information matrix and KL divergence to define the mvp (and
    # dist_fun) used in SecondOrderUpdate
    class decorated_cls(cls, rlPiccoloFisher):

        def __init__(self, policy, base_alg, p=0.0, use_shift=True, damping=0.1, **kwargs):
            assert isinstance(base_alg, BA.SecondOrderUpdate)
            super().__init__(policy, base_alg, p=p, **kwargs)
            # It uses reg_new and reg_old to define the regularization
            # reg_old is intended for memory
            self._reg_old = Reg(self._policy.copy('reg_pol_old'), self._policy, default_damping=damping)
            self._reg_new = Reg(self._policy.copy('reg_pol_new'), self._policy, default_damping=damping)
            self._reg_swp = Reg(self._policy.copy('reg_pol_swp'), self._policy, default_damping=damping)

            # Whether to overide the Fisher with the on-policy one in the
            # Prediction Step. It is not ideal in theory but improves numerical
            # stability.
            self._use_shift = use_shift

        @property
        def _fvp(self):
            return lambda g: (self._reg_new.fvp(g) + self._reg_old.fvp(g)) / 2.0

        @property
        def _kl(self):
            return self._reg_new.kl

        # overwrite _predict and _correct to provde the required mvp and dist_fun
        def _predict(self, g_hat, ro=None, **kwargs):
            obs = self._reg_new.obs if ro is None else ro.obs
            self._reg_swp.update(obs)  # save reg info in swp.
            if self._use_shift:  # may improve numericacl stability but break the theory
                self._reg_old.update(obs)
                self._reg_new.update(obs)
            else:  # use the previous regs, except for initialization
                if not self._reg_old.initialized or not self._reg_new.initialized:
                    self._reg_old.update(obs)
                    self._reg_new.update(obs)

            return super()._predict(g_hat, mvp=self._fvp, **kwargs)

        def _correct(self, g, ro=None, **kwargs):
            # **kwargs are not used
            assert ro is not None
            # update Fisher information matrices
            self._reg_new.update(ro.obs)
            if not self._reg_swp.initialized:  # when no predict has been called.
                self._reg_old.update(ro.obs)
            else:
                self._reg_old.assign(self._reg_swp)

            # call the usual piccolo update
            if isinstance(self._base_alg, BA.TrustRegionSecondOrderUpdate):
                return super()._correct(g, mvp=self._fvp, dist_fun=self._kl, **kwargs)
            elif isinstance(self._base_alg, BA.AdaptiveSecondOrderUpdate):
                return super()._correct(g, mvp=self._fvp, **kwargs)

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


def _rlPiccoloOptDecorator(cls):
    assert issubclass(cls, OO.PiccoloOpt)
    if not issubclass(cls, rlOnlineOptimizer):
        cls = _rlOnlineOptimizerDecorator(cls)

    class decorated_cls(cls):
        def _correct(self, g_hat, grad_hat=None, loss_hat=None, callback=None,
                     warm_start=True, stop_std_grad=False, **kwargs):
            return super()._correct(g_hat, grad_hat=grad_hat, loss_hat=loss_hat, callback=callback, warm_start=warm_start, **kwargs)

        def _predict(self, g_hat, grad_hat=None, loss_hat=None, callback=None,
                     warm_start=True, stop_std_grad=False, **kwargs):
            # grad_hat, loss_hat, callback are functions that take no arguments
            # but inheritently depend on the policy (i.e. self._policy).
            assert (grad_hat is not None) and (loss_hat is not None)

            # prevent std from degenerating during multi-step updates
            if stop_std_grad:
                self._policy.stop_std_grad(True)

                def set_policy(x):
                    std = self._policy.std
                    self._policy.variable = x
                    self._policy.std = std

                def get_projection(x0):
                    # constraint the std as given by x0
                    self._policy.variable = x0
                    std0 = np.copy(self._policy.std)

                    def projection(x):
                        self._policy.variable = x
                        self._policy.std = std0
                        self._policy.stop_std_grad(False)  # turn it back on
                        return self._policy.variable
                    return projection
            else:
                def set_policy(x):
                    self._policy.variable = x
                get_projection = None

            # define the problem
            def _grad_hat(x):
                set_policy(x)
                return grad_hat()

            def _loss_hat(x):
                set_policy(x)
                return loss_hat()
            if callback is not None:
                def _callback(x):
                    self._policy.variable = x
                    return callback()
            else:
                _callback = None
            # call the core library
            g_hat = super()._predict(g_hat, grad_hat=_grad_hat, loss_hat=_loss_hat,
                                     callback=_callback, warm_start=warm_start,
                                     get_projection=get_projection, **kwargs)
            self._policy.stop_std_grad(False)  # turn it back on
            return g_hat

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


# =====================================================================================
# Here we define all rl-oriented classes for policy optimization
@_rlOnlineOptimizerDecorator
class BasicOnlineOptimizer(OO.BasicOnlineOptimizer):
    pass


@_rlOnlineOptimizerDecorator
class Piccolo(OO.Piccolo):
    pass


@_rlPiccoloOptDecorator
@_rlOnlineOptimizerDecorator
class PiccoloOptBasic(OO.PiccoloOptBasic):
    pass


@_rlPiccoloFisherDecorator
class PiccoloFisherReg(Piccolo):
    pass


@_rlPiccoloFisherDecorator
class PiccoloOptBasicFisherReg(PiccoloOptBasic):
    pass

