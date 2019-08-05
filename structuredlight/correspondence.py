import numpy as np
import os


def _reduce_map_columns(pixel_array):
    if pixel_array.dtype == object:
        if any(x.size > 0 for x in pixel_array):
            return np.hstack([x for x in pixel_array.flatten() if x.size != 0])
        else:
            return np.array([])
    return pixel_array


def map_columns(left, right, left_mask, right_mask, keep_unsure=False):
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return np.empty((2, 0))
    # To conserve computation size, we only take a non-masked continuous subset
    nonzero0 = np.stack(left_mask.nonzero())
    nonzero1 = np.stack(right_mask.nonzero())
    b0, e0 = nonzero0.min(axis=1), nonzero0.max(axis=1) + 1
    b1, e1 = nonzero1.min(axis=1), nonzero1.max(axis=1) + 1
    # Note that any dimension but the last must match
    b = tuple(np.max([b0[:-1], b1[:-1]], axis=0).tolist())
    e = tuple(np.min([e0[:-1], e1[:-1]], axis=0).tolist())
    b0, e0 = b + (b0[-1],), e + (e0[-1],)
    b1, e1 = b + (b1[-1],), e + (e1[-1],)
    nd_slice0 = tuple(slice(b, e) for b, e in zip(b0, e0))
    nd_slice1 = tuple(slice(b, e) for b, e in zip(b1, e1))
    left_mask, right_mask = left_mask[nd_slice0], right_mask[nd_slice1]
    left, right = left[nd_slice0], right[nd_slice1]

    # both the left and right side of a match must exist
    match = np.logical_and(left_mask[..., None], right_mask[..., None, :-1])
    match = np.logical_and(match, right_mask[..., None, 1:])

    # value in left should lie between to neighbouring values in right
    match[left[..., :, None] < right[..., None, :-1]] = 0
    match[left[..., :, None] >= right[..., None, 1:]] = 0

    if match.sum() == 0:
        return np.empty((2, 0))

    # Ideally, we should find 1 match between left and right.
    # Since the world isn't ideal, we need to take care of multiple matches.
    unsure = (match.sum(axis=-1) > 1)  # <-- places with multiple matches
    if not keep_unsure:
        # Simply discard all places with multiple matches
        unsure = np.broadcast_to(unsure[..., None], match.shape)
        match[unsure] = 0
        unsure = False

    if np.any(unsure):
        print('Pixels with multiple matches (count):', unsure.sum())

    *index, c0, c1 = tuple(match.nonzero())
    step = right[(*index, c1 + 1)] - right[(*index, c1)]
    c1frac = (left[(*index, c0)] - right[(*index, c1)]) / step

    index = [i + _b for i, _b in zip(index, b)]
    index0 = index + [c0 + b0[-1]]
    index1 = index + [c1 + c1frac + b1[-1]]
    return np.swapaxes(np.array([index0, index1]), 1, 2)
