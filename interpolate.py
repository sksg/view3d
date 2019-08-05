import numpy as np
import os
import cv2


def _bilinear(image, i, j):
    step = 32766  # opencv limit!
    out = np.empty(i.shape + image.shape[-1:], dtype=image.dtype)
    for n in range(0, i.shape[0], step):
        for m in range(0, i.shape[1], step):
            _i = i[n:n + step, m:m + step]
            _j = j[n:n + step, m:m + step]
            im = cv2.remap(image, _j, _i, cv2.INTER_LINEAR)
            shape = out[n:n + step, m:m + step].shape
            out[n:n + step, m:m + step] = im.reshape(shape)
    return np.atleast_3d(out)


def bilinear(image, i, j):
    _i = np.atleast_2d(i).astype(np.float32)
    _j = np.atleast_2d(j).astype(np.float32)
    shape = image.shape[:-3]
    shape = shape if len(shape) > len(_i.shape[:-2]) else _i.shape[:-2]
    shape = shape if len(shape) > len(_j.shape[:-2]) else _j.shape[:-2]
    image = np.broadcast_to(image, shape + image.shape[-3:])
    _i = np.broadcast_to(i, shape + _i.shape[-2:]).astype(np.float32)
    _j = np.broadcast_to(j, shape + _j.shape[-2:]).astype(np.float32)
    out = np.empty(_i.shape + image.shape[-1:], dtype=image.dtype)
    for idx in np.ndindex(_i.shape[:-2]):
        out[idx] = _bilinear(image[idx], _i[idx], _j[idx])
    return out.reshape((*out.shape[:-3], *i.shape[-2:], out.shape[-1]))


def _cubic_rescale(image, H, W):
    return np.atleast_3d(cv2.resize(image, (W, H), cv2.INTER_CUBIC))


def _bilinear_rescale(image, H, W):
    return np.atleast_3d(cv2.resize(image, (W, H), cv2.INTER_LINEAR))


def cubic_rescale(image, scale, scale_W=None):
    if scale_W is None:
        scale_W = scale
    H, W, C = image.shape[-3:]
    H, W = int(scale * H), int(scale_W * W)
    out = np.empty((*image.shape[:-3], H, W, C), image.dtype)
    for idx in np.ndindex(image.shape[:-3]):
        out[idx] = _cubic_rescale(image[idx], H, W)
    return out


def bilinear_rescale(image, scale, scale_W=None):
    if scale_W is None:
        scale_W = scale
    H, W, C = image.shape[-3:]
    H, W = int(scale * H), int(scale_W * W)
    out = np.empty((*image.shape[:-3], H, W, C), image.dtype)
    for idx in np.ndindex(image.shape[:-3]):
        out[idx] = _bilinear_rescale(image, H, W)
    return out
