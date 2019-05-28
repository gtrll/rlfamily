import numpy as np
import tensorflow as tf
from abc import abstractmethod
from rl.tools.utils import tf_utils as U
from rl.tools.utils.misc_utils import zipsame, flatten
from rl.tools.normalizers.normalizer import NormalizerStd
from rl.tools.oracles.oracle import Oracle

tfObject = U.tfObject
tf_float = U.tf_float


class tfOracle(tfObject, Oracle):
    """ An Oracle based on tensorflow.

        Its __deepcopy__ is inherited from tfObject which only deep copies
        attributes in _pre_deepcopy_list and _post_deepcopy_list. Thus, the
        child classes of tfOracle by default does not perform any copying but
        only return a newly instantiated instance according to the original
        arguments in __init__. See tfObject for further instructions on how to
        specify these two lists to deep copy stateful attributes during
        __deepcopy__ call.
    """

    @tfObject.save_init_args()
    def __init__(self, ts_vars, bg_kwargs=None):
        """ Build the graph. """
        self._ts_vars = ts_vars  # a list tf.Variables
        self._args = None  # arguments for evaluating the loss function/gradient
        tfObject.__init__(self, name='tfOracle', bg_kwargs=bg_kwargs)  # build graph

    @property
    def _pre_deepcopy_list(self):
        return super()._pre_deepcopy_list + ['_args']

    def _build_graph(self, **bg_kwargs):
        ts_loss, ph_args = self._build_loss_op(**bg_kwargs)
        # define compute_loss and compute_grad wrt loss
        self._compute_loss = U.function(ph_args, ts_loss)
        ts_grads = U.gradients(ts_loss, self._ts_vars)
        # fill None with zeros; otherwise tf.run will attempt to fetch for None.
        ts_grads = [g if g is not None else tf.zeros_like(v) for (v, g) in
                    zipsame(self._ts_vars, ts_grads)]
        self._compute_grad = U.function(ph_args, ts_grads)

    def compute_loss(self):
        if self._args is None:
            raise ValueError('Oracle has not been initialized')
        return self._compute_loss(*self._args)

    def compute_grad(self):
        if self._args is None:
            raise ValueError('Oracle has not been initialized')
        grads = self._compute_grad(*self._args)
        return flatten(grads)

    def update(self, *args):
        """ Update the data for specifying the placeholders. """
        self._args = args

    @abstractmethod
    def _build_loss_op(self):
        """ Return the loss function as tf.Tensor and a list of tf.placeholders
        required to evaluate the loss function. """


class tfLikelihoodRatioOracle(tfOracle):
    """
    An Oracle based on the loss function below: if use_log_loss is True

        E_{ob} E_{ac ~ q | ob} [ w * log p(ac|ob) * f(ob, ac) ]

    otherwise, it uses

        E_{ob} E_{ac ~ q | ob} [ p(ac|ob)/q(ac|ob) * f(ob, ac) ]

    where p is the variable distribution, q is a constant
    distribution, and f is a scalar function.

    When w = p/q, then the gradients of two loss functions are equivalent.

    The expectation is approximated by unbiased samples from q. To minimize
    the variance of sampled gradients, the implementation of 'compute_grad' is
    based on a normalizer, which can shift, rescale, or clip f.

    """
    @tfObject.save_init_args()
    def __init__(self, ts_vars, ts_logp, ph_args=None, nor=None, correlated=False,
                 use_log_loss=False, normalize_weighting=False):
        # ts_vars: a list of tf.Variables
        # ts_logp (tf.Tensor): the log probability which is a funciton of var
        # ph_args: a list of tf.placeholders necessary to evaluate ts_logp
        if ph_args is None:
            ph_args = []
        assert type(ts_vars) is list
        assert type(ph_args) is list
        self._correlated = correlated
        self._use_log_loss = use_log_loss
        self._normalize_weighting = normalize_weighting
        self._nor = NormalizerStd(None, unscale=True, clip_thre=None,
                                  momentum=0.0) if nor is None else nor  # use mean as control variate
        bg_kwargs = {'ts_logp': ts_logp, 'ph_args': ph_args}
        super().__init__(ts_vars, bg_kwargs=bg_kwargs)  # build graph

    def _build_loss_op(self, ts_logp=None, ph_args=None):
        """ Return the loss function as tf.Tensor and a list of tf.placeholders
        required to evaluate the loss function. """
        assert (not ts_logp is None) and (not ph_args is None)
        ph_f = tf.placeholder(shape=[None], name='function', dtype=tf_float)
        ph_w_or_logq = tf.placeholder(shape=[None], name='w_or_logq', dtype=tf_float)

        # average over samples
        if self._use_log_loss is True:
            ts_loss = tf.reduce_mean(ph_w_or_logq * ph_f * ts_logp)
            if self._normalize_weighting:
                ts_loss = ts_loss / tf.reduce_mean(ph_w_or_logq)

        elif self._use_log_loss is False:

            w = tf.exp(ts_logp - ph_w_or_logq)
            ts_loss = tf.reduce_mean(w * ph_f)
            if self._normalize_weighting:
                ts_loss = ts_loss / tf.reduce_mean(w)

        elif self._use_log_loss is None:
            w = tf.stop_gradient(tf.exp(ts_logp - ph_w_or_logq))
            ts_loss = tf.reduce_mean(w * ph_f * ts_logp)
            if self._normalize_weighting:
                ts_loss = ts_loss / tf.reduce_mean(w)
        else:
            raise ValueError('unknown use_log_loss.')

        ph_args = ph_args + [ph_f, ph_w_or_logq]
        return ts_loss, ph_args

    @property
    def _pre_deepcopy_list(self):
        return super()._pre_deepcopy_list + ['_nor']

    def update(self, f, w_or_logq, arg_list, update_nor=True):
        """
        Update the arguments for the placeholders.

        Args:
            args_list: list of np.ndarrays necessary to evaluate ts_logp.
            f, w (np.ndarrays): values for the tf.placeholders
            update_nor (bool): whether to update the normalizer using the current sample
        """
        if self._correlated and update_nor:
            self._nor.update(f)

        f_normalized = self._nor.normalize(f)  # np.ndarray

        if self._use_log_loss:
            assert np.all(w_or_logq >= 0)
        arg_list = arg_list + [f_normalized, w_or_logq]
        super().update(*arg_list)
        if not self._correlated and update_nor:
            self._nor.update(f)
