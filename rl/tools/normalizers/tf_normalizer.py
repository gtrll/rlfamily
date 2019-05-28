import tensorflow as tf
import numpy as np
import copy
from abc import ABC, abstractmethod

from rl.tools.utils import tf_utils as U
from rl.tools.normalizers.normalizer import OnlineNormalizer, NormalizerStd, NormalizerMax, NormalizerId
from rl.tools.utils.misc_utils import str_to_class, deepcopy_from_list
tfObject = U.tfObject
tf_float = tf.float32


class tfNormalizer:
    @abstractmethod
    def build_nor_ops(self, ts_x):
        """ Return the normalized tensor """


def _tfNormalizerDecorator(cls):
    """ A decorator to reuse the Normalizer classes by building a tensorflow
    graph equivalence of OnlineNormalizer.normalize """
    assert issubclass(cls, OnlineNormalizer)

    class _tfNormalizerParams(tfObject):
        """ a wrapper for maintaining the tf variables """
        @tfObject.save_init_args()
        def __init__(self, name, shape):
            super().__init__(name=name, bg_kwargs={'shape': shape})

        def _build_graph(self, shape):
            self._ts_bias = tf.Variable(dtype=tf_float, initial_value=np.zeros(shape),
                                        name="bias", trainable=False)
            self._ts_scale = tf.Variable(dtype=tf_float, initial_value=np.ones(shape),
                                         name="scale", trainable=False)
            self._ts_initialized = tf.Variable(dtype=tf.bool, initial_value=False,
                                               name='initialized', trainable=False)
            # setter
            self.set_nor_params = U.build_set([self._ts_bias, self._ts_scale, self._ts_initialized])

    class decorated_cls(cls, tfNormalizer):

        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)  # OnlineNormalizer
            self._tf_params = _tfNormalizerParams(cls.__name__, shape)

        def update(self, x):
            super().update(x)
            self._tf_params.set_nor_params(self.bias, self.scale, self._initialized)

        def build_nor_ops(self, ts_x):
            """ build a tf operator that mimics the behavior of self.normalize """
            unscale, unbias, thre = self._unscale, self._unbias, self._clip_thre
            ts_bias, ts_scale = self._tf_params._ts_bias, self._tf_params._ts_scale
            ts_initialized = self._tf_params._ts_initialized

            ts_x0 = tf.identity(ts_x)  # saved for later
            if thre is None:
                if not unbias:
                    ts_x = ts_x - ts_bias
                if not unscale:
                    ts_x = ts_x / ts_scale
            else:
                # need to first rescale so it can clip
                ts_x = (ts_x - ts_bias) / ts_scale
                ts_x = tf.clip_by_value(ts_x, -thre, thre)
                # check if we need to scale it back
                if unscale:
                    ts_x = ts_x * ts_scale
                    if unbias:
                        ts_x = ts_x + ts_bias
                else:
                    if unbias:
                        ts_x = ts_x + self._bias / ts_scale

            # It does nothing in the first iteration
            return U.switch(ts_initialized, ts_x, ts_x0)

        def assign(self, other):
            assert type(self) == type(other)
            copylist = list(self.__dict__.keys())
            copylist.remove('_tf_params')  # do not deepcopy this
            deepcopy_from_list(self, other, copylist)
            self._tf_params.assign(other._tf_params)

        def _debug(self, x):
            ts_nor_x = self.build_nor_ops(x)
            get_ts_val = U.build_get([ts_nor_x])
            v1, v2 = get_ts_val()[0], self.normalize(x0)
            print(v1, v2)
            assert v1 == v2

     # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@_tfNormalizerDecorator
class tfNormalizerId(NormalizerId):
    pass


@_tfNormalizerDecorator
class tfNormalizerStd(NormalizerStd):
    pass


@_tfNormalizerDecorator
class tfNormalizerMax(NormalizerMax):
    pass
