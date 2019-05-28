import collections
import tensorflow as tf
import numpy as np
from functools import wraps
from abc import abstractmethod

from rl.policies.policy import Policy
from rl.tools.function_approximators import tfFunctionApproximator, tfMLPFunctionApproximator, online_compatible
from rl.tools.utils.misc_utils import zipsame
from rl.tools.utils import tf_utils as U
tfObject = U.tfObject
tf_float = U.tf_float


class tfPolicy(tfFunctionApproximator, Policy):
    """
    An abstract helper class to define policies based on tensorflow.

    Note: The seed is for setting the operation-level seed. The randomness is
    tensorflow is determined jointly by graph-level seed and operation-level
    seed. If this operation-level seed is not set, the randomness can be
    different across devices.
    """

    @tfObject.save_init_args()
    def __init__(self, x_dim, y_dim, name='tfPolicy', seed=None,
                 build_nor=None, max_to_keep=None, bg_kwargs=None):

        tfFunctionApproximator.__init__(self, x_dim, y_dim, name=name, seed=seed,
                                        build_nor=build_nor, max_to_keep=max_to_keep,
                                        bg_kwargs=bg_kwargs)
        # so we don't build repetitive operators
        self._kl_cache = collections.defaultdict(lambda: None)

    @property
    def ph_obs(self):
        return self.ph_x

    @property
    def ph_acs(self):
        return self.ph_y

    # required methods of tfFunctionApproximator
    def _build_graph(self, **kwargs):
        """ We treat tfFunctionApproximator as the stochastic map of the policy
        (which inputs ph_x and outputs ts_yh) and build additional
        attributes/methods required by Policy """
        # build tf.Variables
        # add attributes ph_x, ts_nor_x, ts_y, _yh, _sh_vars,
        #                ph_y, ts_pi, ts_logp, ts_pid
        tfFunctionApproximator._build_graph(self, **kwargs)

        # build additional graphs for Policy
        # build conditional distribution
        self._pi = self._yh
        self._pid = U.function([self.ph_x], self.ts_pid)
        self._logp = U.function([self.ph_x, self.ph_y], self.ts_logp)
        # build fvp operator (this depends only on self)
        ph_g, ts_grads = self._sh_vars.build_flat_ph()
        ts_kl = self.build_kl(self, self, p1_sg=True)
        ts_kl_grads = U.gradients(ts_kl, self.ts_vars)  # grad to the 2nd arg of KL
        ts_inner_prod = tf.add_n([tf.reduce_sum(kg * v) for (kg, v) in zipsame(ts_kl_grads, ts_grads)])
        ts_fvp = U.gradients(ts_inner_prod, self.ts_vars)  # Fisher (information matrix) and Vector Product
        ts_fvp = tf.concat([tf.reshape(f, [-1]) for f in ts_fvp], axis=-1)  # continuous vector
        self._fvp = U.function([self.ph_x, ph_g], ts_fvp)

    def _build_func_apprx(self, ts_nor_x, **kwargs):
        # required method called inside tfFunctionApproximator._build_graph
        # add attributes ph_y, ts_pi, ts_logp, ts_pid
        self.ph_y = tf.placeholder(shape=[None, self.y_dim], name="output", dtype=tf_float)
        self.ts_pi, self.ts_logp, self.ts_pid = self._build_dist(ts_nor_x, self.ph_y, **kwargs)  # user-defined
        if self.ts_pid is None:
            self.ts_pid = self.ts_pi
        return self.ts_pi

    # required methods of Policy
    @online_compatible
    def pi(self, x, stochastic=True):
        return self._pi(x) if stochastic else self._pid(x)

    @online_compatible
    def logp(self, x, y):
        return self._logp(x, y)

    def kl(self, other, x, reversesd=False):
        assert type(other) == type(self)
        key = str(id(other)) + str(reversesd)
        if self._kl_cache[key] is None:
            ts_kl = self.build_kl(self, other) if reversesd else self.build_kl(other, self)
            _kl = U.function([self.ph_x, other.ph_x], ts_kl)
            self._kl_cache[key] = lambda _x: _kl(_x, _x)
        return self._kl_cache[key](x)

    def fvp(self, x, g):
        return self._fvp(x, g)

    # methods to be implemented
    @abstractmethod
    def _build_dist(self, ts_x, ph_y, **kwargs):
        """ Return pi, logp, and pid tf.Tensors, which represents the sampled
        action, the derandomized action, and the log probability of a sampled
        action-state pair. If the policy is fully stochastic, pid can be set as
        None.
        """

    @classmethod
    @abstractmethod
    def build_kl(cls, p1, p2, p1_sg=False, p2_sg=False):
        """return KL(p1||p2) as a tf.Tensor, where p1 and p2 are objects of
        this class, and p1_sg and p2_sg denotes whether to treat p1 and p2 as
        constants, respectively, so the gradient is stopped properly when
        compute_gradient is called """


class tfGaussianPolicy(tfPolicy):
    """
    An abstract class which is namely tfPolicy with Gaussian distribution based
    on tfFunctionApproximator.
    """
    @staticmethod
    def _build_logp(dim, rvs, mean, logstd):  # log probability of Gaussian
        return (-0.5 * U.squared_sum((rvs - mean) / tf.exp(logstd), axis=1) -
                tf.reduce_sum(logstd) - 0.5 * np.log(2.0 * np.pi) * tf.to_float(dim))

    @classmethod
    def build_kl(cls, p1, p2, p1_sg=False, p2_sg=False, w=None):
        """KL(p1, p2). p1_sg: whether stop gradient for p1."""
        def get_attr(p, sg):
            logstd, mean = p.ts_logstd, p.ts_mean
            if sg:
                logstd, mean = tf.stop_gradient(logstd), tf.stop_gradient(mean)
            std = tf.exp(logstd)
            return logstd, std, mean
        logstd1, std1, mean1 = get_attr(p1, p1_sg)
        logstd2, std2, mean2 = get_attr(p2, p2_sg)
        kl = logstd2 - logstd1 - 0.5
        kl += (tf.square(std1) + tf.square(mean1 - mean2)) / (2.0 * tf.square(std2))
        kl = tf.reduce_sum(kl, axis=-1)  # reduce over ac dimension
        if w is None:
            kl = tf.reduce_mean(kl)  # average over x
        else:  # weighted average over x
            kl = tf.reduce_sum(kl * w) / tf.reduce_sum(w)
        return kl


def _tfGaussianPolicyDecorator(cls):
    """
    A decorator for defining tfGaussianPolicy via tfFunctionApproximator.
    It is mainly tfGaussianPolicy but uses cls' _build_func_apprx.
    """
    assert issubclass(cls, tfFunctionApproximator)

    class decorated_cls(tfGaussianPolicy, cls):

        @tfObject.save_init_args()
        def __init__(self, x_dim, y_dim, init_logstd,
                     name='tfGaussianPolicy', seed=None,
                     build_nor=None, max_to_keep=None,
                     min_std=0.1, **kwargs):
            # Since tfPolicy only modifies the __init__ of tfFunctionApproximator
            # in adding _kl_cache, we reuse the __init__ of cls.

            # new attributes
            self.ts_mean = self.ts_logstd = self._ts_stop_std_grad = None
            self._init_logstd = init_logstd
            self._min_std = min_std

            # construct a tfFunctionApproximator
            cls.__init__(self, x_dim, y_dim, name=name, seed=seed,
                         build_nor=build_nor, max_to_keep=max_to_keep,
                         **kwargs)

            # add the cache of tfPolicy
            self._kl_cache = collections.defaultdict(lambda: None)

        @property
        def std(self):  # for convenience
            return self._std()

        @std.setter
        def std(self, val):
            self._set_logstd(np.log(val))

        def stop_std_grad(self, cond=True):
            self._set_stop_std_grad(cond)

        def _build_dist(self, ts_nor_x, ph_y):
            # mean and std
            self.ts_mean = cls._build_func_apprx(self, ts_nor_x)  # use the tfFunctionApproximator to define mean
            self._ts_logstd = tf.get_variable(
                'logstd', shape=[self.y_dim], initializer=tf.constant_initializer(self._init_logstd))
            self._ts_stop_std_grad = tf.get_variable('stop_std_grad', initializer=tf.constant(False), trainable=False)
            _ts_logstd = tf.cond(self._ts_stop_std_grad,  # whether to stop gradient
                                 true_fn=lambda: tf.stop_gradient(self._ts_logstd),
                                 false_fn=lambda: self._ts_logstd)
            # make sure the distribution does not degenerate
            self.ts_logstd = tf.maximum(tf.to_float(np.log(self._min_std)), _ts_logstd)
            ts_std = tf.exp(self.ts_logstd)
            self._std = U.function([], ts_std)
            self._set_logstd = U.build_set([self._ts_logstd])
            self._set_stop_std_grad = U.build_set([self._ts_stop_std_grad])

            # pi
            self.ts_noise = tf.random_normal(tf.shape(ts_std), stddev=ts_std, seed=self.seed)
            ts_pi = self.ts_mean + self.ts_noise
            ts_pid = self.ts_mean
            # logp
            ts_logp = self._build_logp(self.y_dim, ph_y, self.ts_mean, self.ts_logstd)
            return ts_pi, ts_logp, ts_pid

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@_tfGaussianPolicyDecorator
class tfGaussianMLPPolicy(tfMLPFunctionApproximator):
    pass
