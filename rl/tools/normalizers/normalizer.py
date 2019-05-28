from abc import ABC, abstractmethod
import numpy as np
import copy
from rl.tools.utils.mvavg import ExpMvAvg, PolMvAvg
from rl.tools.utils.misc_utils import deepcopy_from_list


class OnlineNormalizer(ABC):
    """
        A normalizer that adapts to streaming observations. Given input x, it computes
            x_cooked = clip((x-bias)/scale, thre)
        It should support copy.deepcopy.
    """

    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None):
        self._shape = shape
        self._unscale = unscale
        self._unbias = unbias
        assert clip_thre is None or type(clip_thre) is float, 'Invalid clip_thre.'
        self._clip_thre = clip_thre
        self._initialized = False

    def normalize(self, x):
        """
         Given input x, it computes
            x_cooked = clip((x-bias)/scale, thre)
        If unscale/unbias is True, it removes the scaling/bias after clipping.
        """
        if not self._initialized:
            return x
        # do something
        if self._clip_thre is None:
            if not self._unbias:
                x = x - self.bias
            if not self._unscale:
                x = x / self.scale
        else:
            # need to first scale it before clipping
            x = (x - self.bias) / self.scale
            x = np.clip(x, -self._clip_thre, self._clip_thre)
            # check if we need to scale it back
            if self._unscale:
                x = x * self.scale
                if self._unbias:
                    x = x + self.bias
            else:
                if self._unbias:
                    x = x + self.bias / self.scale
        return x

    def reset(self):
        self._initialized = False
        self._reset()

    @abstractmethod
    def _reset(self):
        """ Reset the normalizer to its initial state. """

    @property
    @abstractmethod
    def bias(self):
        pass

    @property
    @abstractmethod
    def scale(self):
        pass

    def update(self, *args, **kwargs):
        self._update(*args, **kwargs)
        self._initialized = True

    @abstractmethod
    def _update(self, *args, **kwargs):
        """ Update data for defining bias and scale """

    def assign(self, other):
        assert type(self) == type(other)
        deepcopy_from_list(self, other, self.__dict__.keys())


class NormalizerId(OnlineNormalizer):
    # Just an identity map
    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None, **kwargs):
        super().__init__(shape, unscale=True, unbias=True, clip_thre=None)

    def _reset(self):
        pass

    def bias(self):
        return np.zeros(self._shape)

    def scale(self):
        return np.ones(self._shape)

    def _update(self):
        pass


class NormalizerStd(OnlineNormalizer):

    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None,
                 rate=0, momentum=None, eps=1e-6):
        """
            An online normalizer based on whitening.

            shape: None or an tuple specifying each dimension
            momentum: None for moving average
                      [0,1) for expoential average
                      1 for using instant update
            rate: decides the weight of new observation as itr**rate
        """

        super().__init__(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre)
        if momentum is None:
            self._mvavg_init = lambda: PolMvAvg(np.zeros(self._shape), power=rate)
        else:
            assert momentum <= 1.0 and momentum >= 0.0
            self._mvavg_init = lambda: ExpMvAvg(np.zeros(self._shape), rate=momentum)

        self.reset()
        self._eps = eps

    def _reset(self):
        self._mean = self._mvavg_init()
        self._mean_of_sq = self._mvavg_init()

    @property
    def bias(self):
        return self._mean.val

    @property
    def scale(self):
        return np.maximum(self.std, self._eps)

    @property
    def std(self):
        variance = self._mean_of_sq.val - np.square(self._mean.val)
        return np.sqrt(variance)

    def _update(self, x):
        if np.shape(x) == ():
            x = np.array(x)[np.newaxis]
        # observed stats
        new_mean = np.mean(x, axis=0)
        new_mean_of_sq = np.mean(np.square(x), axis=0)
        self._mean.update(new_mean)
        self._mean_of_sq.update(new_mean_of_sq)


class NormalizerMax(OnlineNormalizer):

    def __init__(self, shape, unscale=False, unbias=False, clip_thre=None,
                 rate=0, momentum=None, eps=1e-6):
        # Args:
        #   momentum: None for moving average
        #             [0,1) for expoential average
        #             1 for using instant update
        #   rate: decide the weight of new observation as itr**rate
        super().__init__(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre)
        self._norstd = NormalizerStd(shape, unscale=unscale, unbias=unbias, clip_thre=clip_thre,
                                     rate=rate, momentum=momentum, eps=eps)
        self.reset()
        self._eps = eps

    def _reset(self):
        self._norstd.reset()
        self._upper_bound = None
        self._lower_bound = None

    @property
    def bias(self):
        return 0.5 * self._upper_bound + 0.5 * self._lower_bound

    @property
    def scale(self):
        return np.maximum(self._upper_bound - self.bias, self._eps)

    def _update(self, x):
        # update stats
        self._norstd.update(x)
        # update clipping
        scale_candidate = self._norstd.std
        upper_bound_candidate = self._norstd.bias + self._norstd.scale
        lower_bound_candidate = self._norstd.bias - self._norstd.scale

        if not self._initialized:
            self._upper_bound = upper_bound_candidate
            self._lower_bound = lower_bound_candidate
        else:
            self._upper_bound = np.maximum(self._upper_bound, upper_bound_candidate)
            self._lower_bound = np.minimum(self._lower_bound, lower_bound_candidate)
