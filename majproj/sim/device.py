"""VL53L5CX 8x8 ToF model, matching how the DeepSonar viewer interprets the sensor.

Zone geometry and back-projection follow VL53L5CX-BNO08X-viewer (viewer/config.py, viewer/geometry.py,
MIT, (c) 2026 Henrique Ferrolho): ST per-zone angle tables, perpendicular distance, ST_LOOKUP method.
"""
import numpy as np

RESOLUTION = 8
NUM_ZONES = 64
FOV_DIAGONAL_DEG = 63.0
MIN_RANGE_MM, MAX_RANGE_MM = 20, 4000

# ST official optical calibration tables (viewer/config.py). "pitch" is the elevation from the sensor
# plane (90 deg = straight ahead); the ray z-component is sin(pitch).
ST_PITCH_DEG = np.array([
    59.00, 64.00, 67.50, 70.00, 70.00, 67.50, 64.00, 59.00,
    64.00, 70.00, 72.90, 74.90, 74.90, 72.90, 70.00, 64.00,
    67.50, 72.90, 77.40, 80.50, 80.50, 77.40, 72.90, 67.50,
    70.00, 74.90, 80.50, 85.75, 85.75, 80.50, 74.90, 70.00,
    70.00, 74.90, 80.50, 85.75, 85.75, 80.50, 74.90, 70.00,
    67.50, 72.90, 77.40, 80.50, 80.50, 77.40, 72.90, 67.50,
    64.00, 70.00, 72.90, 74.90, 74.90, 72.90, 70.00, 64.00,
    59.00, 64.00, 67.50, 70.00, 70.00, 67.50, 64.00, 59.00,
])
ST_YAW_DEG = np.array([
    135.00, 125.40, 113.20, 98.13, 81.87, 66.80, 54.60, 45.00,
    144.60, 135.00, 120.96, 101.31, 78.69, 59.04, 45.00, 35.40,
    156.80, 149.04, 135.00, 108.45, 71.55, 45.00, 30.96, 23.20,
    171.87, 168.69, 161.55, 135.00, 45.00, 18.45, 11.31, 8.13,
    188.13, 191.31, 198.45, 225.00, 315.00, 341.55, 348.69, 351.87,
    203.20, 210.96, 225.00, 251.55, 288.45, 315.00, 329.04, 336.80,
    215.40, 225.00, 239.04, 258.69, 281.31, 300.96, 315.00, 324.60,
    225.00, 234.60, 246.80, 261.87, 278.13, 293.20, 305.40, 315.00,
])
_P, _Y = np.deg2rad(ST_PITCH_DEG), np.deg2rad(ST_YAW_DEG)
# Zone centre ray directions in the sensor frame, exactly as the viewer computes them (x negated).
ZONE_DIRS = np.stack([-np.cos(_Y) * np.cos(_P), np.sin(_Y) * np.cos(_P), np.sin(_P)], 1)
ZONE_DIRS /= np.linalg.norm(ZONE_DIRS, axis=1, keepdims=True)
ZONE_PITCH_SIN = np.sin(_P)
ZONE_HALF_WIDTH = np.deg2rad(FOV_DIAGONAL_DEG / np.sqrt(2) / RESOLUTION / 2)  # ~2.8 deg


def backproject(d_perp_mm):
    """Viewer ST_LOOKUP back-projection: 64 perpendicular distances (mm) -> 64 x 3 points (m)."""
    hyp_m = d_perp_mm / ZONE_PITCH_SIN / 1000.0
    p = ZONE_DIRS * hyp_m[:, None]
    p[:, 2] = d_perp_mm / 1000.0
    return p


def subray_dirs(n_side=6):
    """For each zone, n_side^2 directions spread uniformly over its angular footprint -> (64, n, 3)."""
    u = (np.arange(n_side) + 0.5) / n_side * 2 - 1
    du, dv = [g.ravel() * ZONE_HALF_WIDTH for g in np.meshgrid(u, u)]
    c = ZONE_DIRS
    e1 = np.cross(c, [0.0, 1.0, 0.0]); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(e1, c)
    d = c[:, None, :] + np.tan(du)[None, :, None] * e1[:, None, :] + np.tan(dv)[None, :, None] * e2[:, None, :]
    return d / np.linalg.norm(d, axis=2, keepdims=True)


def measure_batch(ranges_m, weights, rng, pulse_sigma_m=0.03, noise_rel=0.01, noise_abs_m=0.003,
                  signal_threshold=0.02, dropout=0.01, n_iter=3):
    """Per-zone ToF return for a batch of frames, vectorized over frames, zones and sub-rays.

    ranges_m, weights: (F, 64, n) radial hit distance and return strength per sub-ray (weight 0 = no return).
    The return histogram is the strength-weighted ranges blurred by the pulse width; the reported range
    is its strongest peak (one target per zone, as the firmware is configured). The peak is found by
    starting from the strongest sub-ray cluster and mean-shifting to the exact mode of the blurred mixture,
    so surfaces closer than the pulse width merge into one peak at an intermediate range (mixed pixels)
    while far-apart surfaces resolve to the stronger one.
    Returns (perpendicular distance in whole mm, status): status 5 = valid like the real sensor. Zones with
    weak signal, range outside [MIN_RANGE_MM, MAX_RANGE_MM] or a random dropout are invalid (status 255, d=0).
    """
    F, Z, n = ranges_m.shape
    r, w = ranges_m, weights
    wsum = w.sum(2)
    ok = wsum >= signal_threshold * n
    peak = (w * r).sum(2) / np.maximum(wsum, 1e-30)                       # single-surface zones: weighted mean
    hit = w > 0
    spread = np.where(hit, r, -np.inf).max(2) - np.where(hit, r, np.inf).min(2)
    cf, cz = np.nonzero(ok & (spread > 0.5 * pulse_sigma_m))             # zones with several surfaces / edges
    if len(cf):
        rc, wc = r[cf, cz], w[cf, cz]                                    # (M, n)
        diff = (rc[:, :, None] - rc[:, None, :]) / pulse_sigma_m         # candidate j vs sample i
        score = (np.exp(-0.5 * diff * diff) * wc[:, None, :]).sum(2)
        peak[cf, cz] = np.take_along_axis(rc, score.argmax(1)[:, None], 1)[:, 0]
    for _ in range(n_iter):
        k = np.exp(-0.5 * ((peak[..., None] - r) / pulse_sigma_m) ** 2) * w
        peak = (k * r).sum(2) / np.maximum(k.sum(2), 1e-30)
    peak = peak + rng.normal(0.0, 1.0, peak.shape) * (noise_abs_m + noise_rel * peak)
    mm = np.round(peak * ZONE_PITCH_SIN[None, :] * 1000.0)                # the sensor reports integer mm
    valid = ok & (mm >= MIN_RANGE_MM) & (mm <= MAX_RANGE_MM) & (rng.random(mm.shape) >= dropout)
    return np.where(valid, mm, 0.0), np.where(valid, 5, 255)


def measure(ranges_m, weights, rng, **kw):
    """Single-frame wrapper of measure_batch: (64, n) inputs -> (64,) perpendicular mm and status."""
    d, s = measure_batch(ranges_m[None], weights[None], rng, **kw)
    return d[0], s[0]


def backproject_batch(d_perp_mm):
    """Vectorized ST_LOOKUP back-projection: (..., 64) perpendicular mm -> (..., 64, 3) points (m)."""
    hyp_m = d_perp_mm / ZONE_PITCH_SIN / 1000.0
    p = ZONE_DIRS * hyp_m[..., None]
    p[..., 2] = d_perp_mm / 1000.0
    return p
