import tensorflow as tf
import numpy as np

from rl.tools.supervised_learners.supervised_learner import SupervisedLearner, predict_in_batches
from rl.tools.function_approximators import tfFunctionApproximator, tfMLPFunctionApproximator

from rl.tools.utils import logz
from rl.tools.utils import minibatch_utils
from rl.tools.utils import math_utils
from rl.tools.utils import tf_utils as U
tfObject = U.tfObject
tf_float = U.tf_float


class tfSupervisedLearner(SupervisedLearner):
    pass


def _tfSupervisedLearnerDecorator(cls):
    """
    A decorator to make tfFunctionApproximator classes self-learnable through
    supervised leanring based on mini-batched optimization (default: Adam).
    """
    assert issubclass(cls, tfFunctionApproximator)

    class decorated_cls(cls, tfSupervisedLearner):

        @tfObject.save_init_args()
        def __init__(self, x_dim, y_dim, name=None, seed=None, build_nor=None, max_to_keep=None,  # for tfFunctionApproximator
                     use_aggregation=False, max_n_samples=None, batch_size_for_prediction=2048,  # for SupervisedLearner
                     batch_size=128, n_batches=2048, learning_rate=1e-3,  # for tfSupervisedLearner
                     **kwargs):  # other arguments for cls.__init__

            if name is None:
                name = cls.__name__
            # just an abstract interface
            tfSupervisedLearner.__init__(self, x_dim, y_dim, name=name, seed=seed,
                                         use_aggregation=use_aggregation,
                                         max_n_samples=max_n_samples,
                                         batch_size_for_prediction=batch_size_for_prediction)
            # build graph
            # NOTE this calls FunctionApproximator.__init__ again but it should be okay
            cls.__init__(self, x_dim, y_dim, name=name, seed=seed,
                         build_nor=build_nor, max_to_keep=max_to_keep, **kwargs)

            # for minibatch optimization
            self.batch_size = batch_size  # size of minibatch
            self.n_batches = n_batches  # maximal number of minibatches used in update
            self.lr = learning_rate

        # Methods of tfFunctionApproximator
        def _build_graph(self, **kwargs):
            """ Add attributes ph_y, ph_w, ph_lr
                    methods _compute_loss, _apply_gradients
            """
            # build function approximator
            cls._build_graph(self, **kwargs)
            # build loss function
            self.ph_y = tf.placeholder(shape=[None, self.y_dim], name="y", dtype=tf_float)
            self.ph_w = tf.placeholder(shape=[None], name='w', dtype=tf_float)  # the weighting for each sample
            ts_loss = self._build_loss(self.ts_yh, self.ph_y, self.ph_w)  # user-defined
            # build optimizer from loss
            ts_grads = list(zip(U.gradients(ts_loss, self.ts_vars), self.ts_vars))  # a list of (grad, var) tuples
            self.ph_lr = tf.placeholder(shape=[], name="learning_rate", dtype=tf_float)
            ts_apply_gradients = self._build_apply_gradients(ts_grads, self.ph_lr)
            self._compute_loss = U.function([self.ph_x, self.ph_y, self.ph_w], ts_loss)
            self._apply_gradients = U.function([self.ph_x, self.ph_y, self.ph_w, self.ph_lr], ts_apply_gradients)

        @predict_in_batches
        def predict(self, x):
            return cls.predict(self, x)

        # Methods of SupervisedLearner
        def _update_func_approx(self, x, y, w, to_log=False, log_prefix=''):
            """ Update the function approximator based on the current data (x, y,
            w) or through self._agg_data which is up-to-date with (x, y, w). """
            # initial loss
            loss_before = self._compute_loss(x, y, w)  # just on the current sample?
            explained_variance_before = math_utils.compute_explained_variance(self.predict(x), y)
            # optimization
            self.prepare_for_update(x)
            x_agg, y_agg, w_agg = self._agg_data['x'], self._agg_data['y'], self._agg_data['w']
            lr = self._update_with_lr_search(x_agg, y_agg, w_agg)  # using aggregated data to update
            # new loss
            loss_after = self._compute_loss(x, y, w)
            explained_variance_after = math_utils.compute_explained_variance(self.predict(x), y)
            if to_log:
                logz.log_tabular('LossBefore({}){}'.format(self.name, log_prefix), loss_before)
                logz.log_tabular('LossAfter({}){}'.format(self.name, log_prefix), loss_after)
                logz.log_tabular('ExplainedVarianceBefore({}){}'.format(self.name, log_prefix), explained_variance_before)
                logz.log_tabular('ExplainedVarianceAfter({}){}'.format(self.name, log_prefix), explained_variance_after)
                logz.log_tabular('UsedLearningRate({}){}'.format(self.name, log_prefix), lr)

        def _update_with_lr_search(self, x, y, w, decay=0.5, maxitr=1, reltol=1e-5):
            # basic optimization routine
            def update_via_minibatches(x, y, w, lr, batch_size, n_batches):
                n_batches_so_far = 0
                done = False
                while not done:
                    for (batch_x, batch_y, batch_w) in minibatch_utils.generate_batches(
                            (x, y, w), include_final_partial_batch=True, batch_size=batch_size, shuffle=True):
                        self._apply_gradients(batch_x, batch_y, batch_w, lr)
                        n_batches_so_far += 1
                        if n_batches_so_far >= n_batches:
                            done = True
            # find a lr that decreases the loss function after optimization
            lr = self.lr
            variable0 = self.variable
            loss0 = self._compute_loss(x, y, w)
            itr = 0
            while True:
                update_via_minibatches(x, y, w, lr, self.batch_size, self.n_batches)
                # check if the loss is decreased after the update
                if (self._compute_loss(x, y, w) - loss0) / np.abs(loss0 + 1e-8) <= reltol:
                    break
                else:  # try a smaller step size
                    self.variable = variable0
                    lr *= decay
                    itr += 1
                    if itr > maxitr:
                        print('Reached the maximal number of {} tries with lr {}. Reset the value to the original value.'.format(maxitr, lr))
                        break
            return lr  # the step size used at the end

        # Methods that can be overloaded
        def _build_apply_gradients(self, ts_grads, ts_lr):
            """ This can be overloaded """
            return tf.train.AdamOptimizer(ts_lr).apply_gradients(ts_grads)

        def _build_loss(self, ts_yh, ts_y, ts_w):
            """ Define the loss for optimization """
            ts_loss = tf.reduce_sum(tf.square(ts_y - ts_yh), axis=1)  # sum over output dim
            return tf.reduce_mean(ts_loss * ts_w)

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@_tfSupervisedLearnerDecorator
class tfMLPSupervisedLearner(tfMLPFunctionApproximator):
    pass
