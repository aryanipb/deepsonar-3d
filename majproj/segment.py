"""Split a single-view scene scan into background (wall + floor) and individual objects."""
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class Segmentation:
    points: np.ndarray                      # N x 3, original scan
    background: np.ndarray                  # bool mask, wall + floor + stray points
    objects: list = field(default_factory=list)  # list of index arrays, largest first


def fit_plane(pts, axis, thresh, iters=400, max_tilt_deg=30, seed=0):
    """RANSAC plane whose normal is within max_tilt_deg of the given axis. Returns (normal, d, inlier mask)."""
    rng = np.random.default_rng(seed)
    cos_min = np.cos(np.deg2rad(max_tilt_deg))
    best = (None, None, np.zeros(len(pts), bool))
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n[axis]) < cos_min:
            continue
        d = -n @ a
        inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    if n is not None and inl.sum() >= 3:  # least-squares refit on inliers
        q = pts[inl]
        c = q.mean(0)
        n = np.linalg.svd(q - c)[2][-1]
        d = -n @ c
        inl = np.abs(pts @ n + d) < thresh
    return n, d, inl


def segment(points, n_objects=3, wall_thresh=0.04, floor_thresh=0.05, eps=0.07, min_cluster=40,
            top_y=0.75, side_margin=0.06):
    pts = np.asarray(points, np.float32)
    bg = np.zeros(len(pts), bool)

    # wall: dominant plane facing the camera (normal ~ z); everything behind it is background too
    n, d, inl = fit_plane(pts, axis=2, thresh=wall_thresh)
    if n is not None:
        n = n if n[2] > 0 else -n
        d = -n @ pts[inl].mean(0)
        bg |= (pts @ n + d) > -wall_thresh

    # floor: plane with normal ~ y among the remaining low points
    low = np.where(~bg & (pts[:, 1] < -0.6))[0]
    if len(low) >= 10:
        n, d, inl = fit_plane(pts[low], axis=1, thresh=floor_thresh)
        if n is not None and inl.sum() >= 10:
            n = n if n[1] > 0 else -n
            d = -n @ pts[low][inl].mean(0)
            bg |= (pts @ n + d) < floor_thresh

    fg = np.where(~bg)[0]
    objects = []
    if len(fg) >= min_cluster:
        labels = DBSCAN(eps=eps, min_samples=8).fit_predict(pts[fg])
        ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        order = ids[np.argsort(-counts)]
        x_min = pts[:, 0].min()

        def is_object(idx):
            # scene-specific: drop the wall's top-edge fragments (floating at y ~ 1) and the
            # side-wall strip at the left scan boundary; the objects stand lower / further right
            q = pts[idx]
            return np.median(q[:, 1]) < top_y and q[:, 0].min() > x_min + side_margin

        objects = [fg[labels == k] for k in order if (labels == k).sum() >= min_cluster]
        objects = [o for o in objects if is_object(o)][:n_objects]
        if not objects:
            objects = [fg]
    keep = np.concatenate(objects) if objects else np.array([], int)
    bg = np.ones(len(pts), bool)
    bg[keep] = False
    return Segmentation(points=pts, background=bg, objects=objects)


def find_target(points, eps=0.04, max_size=0.3, min_points=100):
    """One scanned object per scan: the densest compact cluster in front of the background.
    Returns (target indices, background mask). The scene furniture (wall, slab, poles) is background."""
    pts = np.asarray(points, np.float32)
    # everything that survived wall/floor removal and lies in the front half of the scene
    cand = np.where(~_wall_floor_mask(pts) & (pts[:, 2] < 0))[0]
    best = None
    if len(cand) >= min_points:
        labels = DBSCAN(eps=eps, min_samples=8).fit_predict(pts[cand])
        for k in np.unique(labels[labels >= 0]):
            idx = cand[labels == k]
            ext = np.percentile(pts[idx], 98, 0) - np.percentile(pts[idx], 2, 0)
            if len(idx) >= min_points and (ext < max_size).all() and (best is None or len(idx) > len(best)):
                best = idx
    if best is None:  # no compact cluster: fall back to the whole front foreground
        best = cand if len(cand) else np.arange(len(pts))
    bg = np.ones(len(pts), bool)
    bg[best] = False
    return best, bg


def _wall_floor_mask(pts, wall_thresh=0.04, floor_thresh=0.05):
    bg = np.zeros(len(pts), bool)
    n, d, inl = fit_plane(pts, axis=2, thresh=wall_thresh)
    if n is not None:
        n = n if n[2] > 0 else -n
        d = -n @ pts[inl].mean(0)
        bg |= (pts @ n + d) > -wall_thresh
    low = np.where(~bg & (pts[:, 1] < -0.6))[0]
    if len(low) >= 10:
        n, d, inl = fit_plane(pts[low], axis=1, thresh=floor_thresh)
        if n is not None and inl.sum() >= 10:
            n = n if n[1] > 0 else -n
            d = -n @ pts[low][inl].mean(0)
            bg |= (pts @ n + d) < floor_thresh
    return bg
