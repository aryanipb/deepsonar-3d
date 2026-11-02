"""Per-object normalization, computed from the partial scan only (identical for training and inference).

Frame: +y up, the sensor looks along +z. ToF depth inside a small object is smeared along the viewing
ray (flying pixels), so the scale comes from the x-y footprint and the depth anchor from the front face."""
import numpy as np

FOOTPRINT = 0.25  # normalized size of the larger x/y footprint side


def resample(pts, n, rng=None):
    """Random resampling (training augmentation)."""
    rng = rng or np.random.default_rng(0)
    return pts[rng.choice(len(pts), n, replace=len(pts) < n)]


def resample_fixed(pts, n):
    """Deterministic resampling: tile then truncate (reproducible in JavaScript)."""
    reps = -(-n // len(pts))
    return np.tile(pts, (reps, 1))[:n]


def footprint(partial):
    lo, hi = np.percentile(partial, 2, 0), np.percentile(partial, 98, 0)
    return lo, hi


def normalize_params(partial):
    """Center = footprint center in x/y and the front-face depth in z; scale from the x/y footprint only."""
    lo, hi = footprint(partial)
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, np.percentile(partial[:, 2], 10)], np.float32)
    scale = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-3)) / FOOTPRINT
    return center, scale


def normalize(pts, center, scale):
    return ((pts - center) / scale).astype(np.float32)


def denormalize(pts, center, scale):
    return (pts * scale + center).astype(np.float32)
