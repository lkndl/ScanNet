"""Keras 2 ``keras.backend`` helpers, on top of the Keras 3 ``keras.ops`` API.

Keras 3 dropped most of the ``keras.backend`` functions this codebase was
written against. Importing this module as ``K`` keeps the original call sites
working: every name is looked up in ``keras.ops`` first, then in the remaining
``keras.backend``, and the few Keras 2 helpers that have no equivalent are
defined below.
"""
from keras import ops
from keras import backend as _keras_backend


def __getattr__(name):
    for module in (ops, _keras_backend):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def int_shape(x):
    return tuple(x.shape)


def dtype(x):
    return _keras_backend.standardize_dtype(x.dtype)


def batch_normalization(x, mean, variance, beta, gamma, axis=-1, epsilon=1e-3):
    inv = ops.rsqrt(variance + epsilon)
    if gamma is not None:
        inv = inv * gamma
    res = -mean * inv
    if beta is not None:
        res = res + beta
    return x * inv + res


def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    mean = ops.mean(x, axis=reduction_axes, keepdims=True)
    variance = ops.var(x, axis=reduction_axes, keepdims=True)
    normed = batch_normalization(x, mean, variance, beta, gamma,
                                 epsilon=epsilon)
    return normed, ops.squeeze(mean, reduction_axes), ops.squeeze(variance, reduction_axes)


def moving_average_update(variable, value, momentum):
    variable.assign(variable * momentum + value * (1.0 - momentum))
    return variable


def in_train_phase(x, alt, training=None):
    if training:
        return x() if callable(x) else x
    return alt() if callable(alt) else alt
