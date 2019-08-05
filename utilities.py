import numpy as np
import os
import cv2


def iround(array, dtype=int):
    return np.rint(array).astype(dtype)


def ifloor(array, dtype=int):
    return np.floor(array).astype(dtype)


def iceil(array, dtype=int):
    return np.ceil(array).astype(dtype)


def ndtake(slicing, axis):  # simple slicing, much faster than np.take
    if axis >= 0:
        return (slice(None),) * axis + (slicing,)
    if axis < 0:
        return (Ellipsis, slicing,) + (slice(None),) * (-1 - axis)


def ndsplit(array, widths, axis=0):  # split axis into len(widths) parts
    splits = (0,) + tuple(np.cumsum(widths))
    return [array[ndtake(slice(s0, s1), axis)]
            if s1 - s0 > 1 else array[ndtake(s0, axis)]
            for s0, s1 in zip(splits[:-1], splits[1:])]


def vectordot(u, v, *args, **kwargs):
    """Specilization of the dot-operator. u and v are ndarrays of vectors"""
    u, v = np.broadcast_arrays(u, v)
    return np.einsum('...i,...i ->...', u, v).reshape(*u.shape[:-1], 1)


def HDR_using_opencv(frames, exposure_times=None):
    list_shape = frames.shape[1:-3]
    if len(list_shape) == 0:
        return HDR_using_opencv(frames[:, None], exposure_times)[0]
    if exposure_times is None:
        exposure_times = np.arange(0, frames.shape[0], dtype=np.float32)
        exposure_times = 255 * np.power(2, exposure_times)
    result = np.empty((frames.shape[1:]), dtype=np.float32)
    merge_debvec = cv2.createMergeDebevec()
    for idx in np.ndindex(list_shape):
        frame = frames[(slice(None),) + idx]
        result[idx] = merge_debvec.process(frame, times=exposure_times.copy())
    return result
