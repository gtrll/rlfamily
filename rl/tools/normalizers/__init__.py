from .normalizer import OnlineNormalizer, NormalizerStd, NormalizerMax, NormalizerId
from .tf_normalizer import tfNormalizer, tfNormalizerMax, tfNormalizerStd, tfNormalizerId


def create_build_nor_from_str(nor_cls_str, nor_kwargs):
    nor_cls = globals()[nor_cls_str]

    def build_nor(shape):
        return nor_cls(shape, unscale=False, unbias=False, **nor_kwargs)
    return build_nor
