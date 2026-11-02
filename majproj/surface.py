"""Turn a completed point cloud into one closed 3D surface.

Stray points are dropped, each horizontal slice is filled with its convex outline, and the outer boundary is extracted
with marching cubes. Only geometry: no template and no object labels, so the surface shows exactly what the model predicted.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from skimage.morphology import convex_hull_image

VOXEL_M = 0.003
MIN_NEARBY = 12                 # points within 2 voxels: fewer than this is a stray point


def to_mesh(points_m, voxel=VOXEL_M):
    """N x 3 metres, z up -> (vertices V x 3 metres, triangles F x 3): the largest closed surface.

    Each horizontal slice is filled with the convex outline of its points. That needs an upright object with convex
    slices, which the cleaner already requires to be upright; a concave cross-section would be filled in."""
    p = np.asarray(points_m, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 100 or not np.isfinite(p).all():
        raise ValueError('expected at least 100 finite N x 3 points')
    pad = 5
    origin = p.min(0) - pad * voxel
    idx = np.floor((p - origin) / voxel).astype(int)
    counts = np.zeros(idx.max(0) + pad + 1, np.float32)
    np.add.at(counts, tuple(idx.T), 1)
    nearby = ndi.uniform_filter(counts, 5, mode='constant') * 125
    grid = (counts > 0) & (nearby >= MIN_NEARBY)
    solid = np.zeros_like(grid)
    for k in range(1, grid.shape[2] - 1):
        layer = grid[:, :, k - 1:k + 2].any(2)
        if layer.sum() >= 3:
            solid[:, :, k] = convex_hull_image(layer)
    labels, n = ndi.label(solid)
    if n == 0:
        raise ValueError('no solid region')
    solid = labels == 1 + int(np.argmax(ndi.sum(solid, labels, range(1, n + 1))))
    v, f, _, _ = marching_cubes(ndi.gaussian_filter(solid.astype(np.float32), 1.5), 0.5)
    return v * voxel + origin + voxel / 2, f.astype(np.int64)


def is_closed(f):
    """True when every edge is shared by exactly two triangles."""
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    return bool((np.unique(e, axis=0, return_counts=True)[1] == 2).all())


def write_obj(path, v, f):
    with open(path, 'w') as o:
        o.writelines(f'v {a:.5f} {b:.5f} {c:.5f}\n' for a, b, c in v)
        o.writelines(f'f {a + 1} {b + 1} {c + 1}\n' for a, b, c in f)
