"""Analytic ray casting for simulated scenes (planes, boxes, vertical solids of revolution).

All primitives return the ray parameter t of the first hit (np.inf on miss) for unit directions, plus
the surface normal used for the return-strength model. Ray origins `o` may be one point (3,) or one per
ray (N, 3). World frame: +y up. `cast` culls rays against each primitive's bounding box first (speed).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray
    reflectivity: float = 0.6

    def unit_normal(self):
        return self.normal / np.linalg.norm(self.normal)

    def hit_t(self, o, d):
        """Ray parameter only (the normal is constant)."""
        n = self.unit_normal()
        den = d @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (self.point @ n - np.asarray(o, float) @ n) / den
        return np.where((np.abs(den) > 1e-9) & (t > 1e-6), t, np.inf)

    def intersect(self, o, d):
        return self.hit_t(o, d), np.broadcast_to(self.unit_normal(), d.shape)


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    reflectivity: float = 0.6

    def bounds(self):
        return np.asarray(self.lo, float), np.asarray(self.hi, float)

    def surface(self, n, rng):
        return box_surface(self, n, rng)

    def intersect(self, o, d):
        o = np.asarray(o, float)
        with np.errstate(divide='ignore', invalid='ignore'):
            t1, t2 = (self.lo - o) / d, (self.hi - o) / d
        tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
        tmin = np.where(np.isnan(tmin), -np.inf, tmin)
        tmax = np.where(np.isnan(tmax), np.inf, tmax)
        tn = np.maximum(np.maximum(tmin[:, 0], tmin[:, 1]), tmin[:, 2])     # per-axis: 5x cheaper than .max(1) on 3 columns
        tf = np.minimum(np.minimum(tmax[:, 0], tmax[:, 1]), tmax[:, 2])
        hit = (tn <= tf) & (tf > 1e-6)
        t = np.where(hit, np.where(tn > 1e-6, tn, tf), np.inf)
        axis = tmin.argmax(1)
        normal = np.zeros_like(d)
        normal[np.arange(len(d)), axis] = -np.sign(d[np.arange(len(d)), axis])
        return t, normal


@dataclass
class Revolution:
    """Solid of revolution around a vertical axis through (cx, cz): piecewise-linear radius profile r(y),
    closed with flat caps. Covers bottles (body + shoulder + neck), cans, jars, cylinders."""
    cx: float
    cz: float
    ys: np.ndarray          # increasing heights
    rs: np.ndarray          # radius at each height
    reflectivity: float = 0.6

    def bounds(self):
        rm = float(np.max(self.rs))
        return (np.array([self.cx - rm, self.ys[0], self.cz - rm]),
                np.array([self.cx + rm, self.ys[-1], self.cz + rm]))

    def intersect(self, o, d):
        best = np.full(len(d), np.inf)
        normal = np.zeros_like(d)
        o = np.broadcast_to(np.asarray(o, float), d.shape)
        ox, oy, oz = o[:, 0] - self.cx, o[:, 1], o[:, 2] - self.cz
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        for y0, y1, r0, r1 in zip(self.ys[:-1], self.ys[1:], self.rs[:-1], self.rs[1:]):
            k = (r1 - r0) / (y1 - y0)            # radius r(y) = r0 + k (y - y0)
            # (ox + t dx)^2 + (oz + t dz)^2 = (r0 + k (oy + t dy - y0))^2
            rb = r0 + k * (oy - y0)
            a = dx * dx + dz * dz - (k * dy) ** 2
            b = 2 * (ox * dx + oz * dz - rb * k * dy)
            c = ox * ox + oz * oz - rb * rb
            disc = b * b - 4 * a * c
            ok = disc >= 0
            sq = np.sqrt(np.where(ok, disc, 0))
            with np.errstate(divide='ignore', invalid='ignore'):
                for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
                    y = oy + t * dy
                    valid = ok & (t > 1e-6) & (y >= y0) & (y <= y1) & (r0 + k * (y - y0) >= 0)
                    upd = valid & (t < best)
                    best = np.where(upd, t, best)
                    px, pz = ox + t * dx, oz + t * dz
                    rr = np.hypot(px, pz) + 1e-12
                    n = np.stack([px / rr, -k * np.ones_like(px), pz / rr], 1)
                    n /= np.linalg.norm(n, axis=1, keepdims=True)
                    normal = np.where(upd[:, None], n, normal)
        for yc, rc, sgn in ((self.ys[0], self.rs[0], -1.0), (self.ys[-1], self.rs[-1], 1.0)):
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (yc - oy) / dy
            px, pz = ox + t * dx, oz + t * dz
            valid = (t > 1e-6) & (px * px + pz * pz <= rc * rc)
            upd = valid & (t < best)
            best = np.where(upd, t, best)
            normal = np.where(upd[:, None], np.array([0.0, sgn, 0.0]), normal)
        return best, normal

    def surface(self, n, rng):
        """Uniform samples on the complete closed surface (ground truth)."""
        ys, rs = np.asarray(self.ys, float), np.asarray(self.rs, float)
        seg = np.pi * (rs[:-1] + rs[1:]) * np.hypot(np.diff(ys), np.diff(rs))
        caps = np.pi * np.array([rs[0] ** 2, rs[-1] ** 2])
        areas = np.concatenate([seg, caps])
        counts = rng.multinomial(n, areas / areas.sum())
        pts = []
        for i, k in enumerate(counts[:-2]):
            u = rng.uniform(0, 1, k)
            r0, r1 = rs[i], rs[i + 1]
            t = u if abs(r1 - r0) < 1e-12 else (np.sqrt(r0 ** 2 + u * (r1 ** 2 - r0 ** 2)) - r0) / (r1 - r0)
            r, y, a = r0 + (r1 - r0) * t, ys[i] + (ys[i + 1] - ys[i]) * t, rng.uniform(0, 2 * np.pi, k)
            pts.append(np.stack([self.cx + r * np.cos(a), y, self.cz + r * np.sin(a)], 1))
        for (yc, rc), k in zip([(ys[0], rs[0]), (ys[-1], rs[-1])], counts[-2:]):
            a, r = rng.uniform(0, 2 * np.pi, k), rc * np.sqrt(rng.uniform(0, 1, k))
            pts.append(np.stack([self.cx + r * np.cos(a), np.full(k, yc), self.cz + r * np.sin(a)], 1))
        return np.concatenate(pts)


def box_surface(box, n, rng):
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    e = hi - lo
    areas = np.array([e[1] * e[2], e[0] * e[2], e[0] * e[1]]).repeat(2)
    counts = rng.multinomial(n, areas / areas.sum())
    pts = []
    for f, k in enumerate(counts):
        axis = f // 2
        p = lo + rng.uniform(0, 1, (k, 3)) * e
        p[:, axis] = hi[axis] if f % 2 else lo[axis]
        pts.append(p)
    return np.concatenate(pts)


@dataclass
class RotatedBox:
    """Box with half-extents `half` centred at `center`, rotated by `yaw` (rad) about the vertical +y axis."""
    center: np.ndarray
    half: np.ndarray
    yaw: float
    reflectivity: float = 0.6

    def rotation(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])        # local -> world

    def bounds(self):
        ext = np.abs(self.rotation()) @ np.asarray(self.half, float)
        c = np.asarray(self.center, float)
        return c - ext, c + ext

    def intersect(self, o, d):
        R = self.rotation()
        c = np.asarray(self.center, float)
        h = np.asarray(self.half, float)
        t, n_local = Box(-h, h, self.reflectivity).intersect((np.asarray(o, float) - c) @ R, d @ R)
        return t, n_local @ R.T

    def surface(self, n, rng):
        h = np.asarray(self.half, float)
        p = box_surface(Box(-h, h), n, rng)
        return p @ self.rotation().T + np.asarray(self.center, float)


def cast(prims, o, d):
    """First hit over all primitives: (t, normal, reflectivity, primitive index).

    Primitives that expose `bounds()` only see the rays that pass through their bounding box."""
    o = np.broadcast_to(np.asarray(o, float), d.shape)
    t_best = np.full(len(d), np.inf)
    n_best = np.zeros_like(d)
    refl = np.zeros(len(d))
    which = np.full(len(d), -1)
    inv = None
    planes = []
    for i, p in enumerate(prims):
        if isinstance(p, Plane):                                  # unbounded: ray parameter only
            t = p.hit_t(o, d)
            upd = t < t_best
            t_best = np.where(upd, t, t_best)
            refl = np.where(upd, p.reflectivity, refl)
            which = np.where(upd, i, which)
            planes.append(i)
            continue
        if hasattr(p, 'bounds'):
            if inv is None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv = 1.0 / d
            lo, hi = p.bounds()
            with np.errstate(invalid='ignore'):
                t1, t2 = (lo - o) * inv, (hi - o) * inv
                tn = np.minimum(t1, t2).max(1)
                tf = np.maximum(t1, t2).min(1)
            idx = np.nonzero((tn <= tf) & (tf > 1e-6))[0]
            if len(idx) == 0:
                continue
            t_s, n_s = p.intersect(o[idx], d[idx])
            upd_s = t_s < t_best[idx]
            sel = idx[upd_s]
            t_best[sel] = t_s[upd_s]
            n_best[sel] = n_s[upd_s]
            refl[sel] = p.reflectivity
            which[sel] = i
            continue
        t, n = p.intersect(o, d)
        upd = t < t_best
        t_best = np.where(upd, t, t_best)
        n_best = np.where(upd[:, None], n, n_best)
        refl = np.where(upd, p.reflectivity, refl)
        which = np.where(upd, i, which)
    for i in planes:
        n_best[which == i] = prims[i].unit_normal()
    return t_best, n_best, refl, which
