import numpy as np
import os
from ..utilities import vectordot

def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def from_depth(camera, points3D, pixels):
    mask = np.zeros(camera.imshape, bool)
    mask[(*(pixels + 0.5).T.astype(int),)] = 1
    xyz = np.full(tuple(camera.imshape) + (3,), np.nan, points3D.dtype)
    xyz[mask] = points3D

    dxyzdx, dxyzdy = np.gradient(xyz, axis=(0, 1))
    dxyzdx, dxyzdy = dxyzdx[mask], dxyzdy[mask]

    n = normalize(np.cross(dxyzdx, dxyzdy))

    nan_count = np.isnan(n).sum()
    if nan_count > 0:
        print('Warning: from_depth() produced {} nans'.format(nan_count))

    c = normalize(camera.position - points3D)
    dev = (n * c).sum(axis=1)
    n *= np.sign(dev)[:, None]
    return n, np.abs(dev)
