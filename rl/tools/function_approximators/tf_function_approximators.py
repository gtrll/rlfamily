import tensorflow as tf
from abc import abstractmethod
from rl.tools.function_approximators.function_approximator import FunctionApproximator, online_compatible

from rl.tools.normalizers import tfNormalizer, tfNormalizerMax
from rl.tools.utils import tf_utils as U
tfObject = U.tfObject
tf_float = U.tf_float


class tfFunctionApproximator(tfObject, FunctionApproximator):
    """
    An abstract helper class to define function approximators based on tensorflow.

    Note: The seed is for setting the operation-level seed. The randomness is
    tensorflow is determined jointly by graph-level seed and operation-level
    seed. If this operation-level seed is not set, the randomness can be
    different across devices.
    """

    @tfObject.save_init_args()
    def __init__(self, x_dim, y_dim, name='tfFunctionApproximator', seed=None,
                 build_nor=None, max_to_keep=None, bg_kwargs=None):

        if bg_kwargs is None:
            bg_kwargs = {}

        # just an abstract interface
        FunctionApproximator.__init__(self, x_dim, y_dim, name=name, seed=seed)
        # build normalizer (an tfObject)
        if build_nor is None:
            self._nor = tfNormalizerMax((self.x_dim,), unscale=False,
                                        unbias=False, clip_thre=5.0, rate=0., momentum=None)
        else:
            self._nor = build_nor((self.x_dim,))
            assert isinstance(self._nor, tfNormalizer)
        # build tf graph
        tfObject.__init__(self, name, max_to_keep=max_to_keep,
                          bg_kwargs=bg_kwargs)

    # Required methods of tfObject
    def _build_graph(self, **kwargs):
        """
        Builds the graph of mapping through the user-provided
        _build_func_apprx.  After all the tf.Variables are created it adds a
        new attribute _sh_vars (a Shaper object) for convenient manipulation of
        the tf.Variables inside the graph.

        Added attributes:
            ph_x, ts_nor_x, ts_y, _yh, _sh_vars
        """
        # build the input placeholder
        self.ph_x = tf.placeholder(shape=[None, self.x_dim], name="input", dtype=tf_float)
        # build the normalizer for whitening
        self.ts_nor_x = self._nor.build_nor_ops(self.ph_x)
        # build parameterized function approximator
        self.ts_yh = self._build_func_apprx(self.ts_nor_x, **kwargs)
        self._yh = U.function([self.ph_x], self.ts_yh)

        # build a Shaper of trainable variables for transforming
        # between continguous and list representations
        self._sh_vars = U.Shaper(self.ts_vars)

    @property
    def _pre_deepcopy_list(self):
        return tfObject._pre_deepcopy_list.fget(self) + ['_nor']

    # Required methods of FunctionApproximator
    # save, restore, copy, __deepcopy__ have been inherited from tfObject
    @online_compatible
    def predict(self, x):
        return self._yh(x)

    def prepare_for_update(self, x):
        self._nor.update(x)

    @property
    def variable(self):
        return self._sh_vars.variable

    @variable.setter
    def variable(self, val):
        self._sh_vars.variable = val

    def assign(self, other):
        # need to overload this because of self._nor
        assert type(self) == type(other)
        tfObject.assign(self, other)
        self._nor.assign(other._nor)

    # Methods to be implemented
    @abstractmethod
    def _build_func_apprx(self, ts_nor_x, **kwargs):
        """ Return the prediction given normalized input ts_nor_x as a
        tf.Tensor. The extra arguments args and kwargs can be passed as
        bg_kwargs in __init__. """


class tfMLPFunctionApproximator(tfFunctionApproximator):

    @tfObject.save_init_args()
    def __init__(self, x_dim, y_dim, n_layers, size, name='tfMLPFunctionApproximator',
                 seed=None, build_nor=None, max_to_keep=None, activation=tf.tanh):
        # setting for MLP
        self._n_layers = n_layers
        self._size = size
        self._activation = activation
        super().__init__(x_dim, y_dim, name=name, seed=seed,
                         build_nor=build_nor, max_to_keep=max_to_keep)

    def _build_func_apprx(self, ts_nor_x):
        return U.build_multilayer_perceptron('yhat', ts_nor_x, self.y_dim, self._n_layers, self._size, self._activation, output_init_std=0.01)
