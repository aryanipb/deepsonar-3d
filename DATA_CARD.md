# Data card

**Status, 2026-11-02.** Real data: 27 raw files, 20 sessions, one box and one bottle, all in `FINAL-INPUTS/`. Training data: simulated
scans built from those sessions with the real object cut out, plus 18 real sessions in the final training stage.

| | Files or scans | Use |
|---|---|---|
| `FINAL-INPUTS/BOX/` | 14 files, 11 sessions | calibration, room bank, evaluation |
| `FINAL-INPUTS/BOTTLE/` | 13 files, 9 sessions | calibration, room bank, evaluation |
| `ds_final` (Modal volume `majproj-vol`) | 64,847 train, 3,675 val, 1,857 test | training and evaluation |

## 1. Raw file format

Nx3 float64, **metres, z = up, sensor at the map origin**, 25k-100k rows. No timestamps, intensities or poses. Measured
(`docs/evidence/real_stats.md`, `scripts/real_stats.py`):

* Append-only, so the row index is a time axis. Repeat exports are byte-exact prefixes.
* Most frames are written twice: doubled blocks cover 92.4 % of rows at median. Some sessions repeat one frame for
  thousands of rows (up to 74,850). Duplicate factor 1.93 at median, 6.80 at worst.
* 8 x 8 geometry check: neighbouring zones on the ceiling are 5.97 deg apart, against 6.00 deg from the ST table.
* No frame has all 64 zones, the largest has 58. While aimed at the object the median is 24 of 64 (box) and 17 (bottle).

Repeat exports of a session (count once, split by session):

| Session | Files, smallest to largest |
|---|---|
| box | 121018 < 121022 < 121055; 121422 < 121457 |
| bottle | 122617 / 122622; 123027 / 123028; 123257 / 123259; 123814 / 123815 |

The older `scan_*_shapenet.npy` exports are 4096 rows, centred and normalized. They carry no metric scale. They are not training data.

## 2. Anatomy of a file

| Layer | Cause | Measured |
|---|---|---|
| ceiling | device lying on the table, facing up | z 2.80 m, 46 (box) and 37 (bottle) valid zones at median |
| mid layer | seen in ceiling frames (98 % of its hits at median) | median z 2.06 m |
| swing streaks | pick-up and put-down | line-like, a few points per zone |
| table | aiming low | sensor 0.094 m (box), 0.070 m (bottle) above it at median |
| background A, B | fixed furniture, median range 0.83 to 0.95 m | A near (-0.40, 0.82), B near (0.16, 0.80); depends on aim |
| near-sensor junk | hand, cover | 0.01 % of distinct points at median |
| depth smear | one-way approach during the dwell, 20 of 20 sessions | see below |
| **object** | held about 25 s at 15 Hz | see below |

The drift cause (hand, ranging or both) cannot be separated from these files.

## 3. The real objects, as measured

Median (range) over sessions, after `thin()` and `clean(origin)`:

| | Box, 11 sessions | Bottle, 9 sessions |
|---|---|---|
| Seen face width | 18.8 cm (14.0 to 21.9) | 10.4 cm (9.5 to 13.1) |
| Seen face height | 23.3 cm (20.9 to 25.5) | 25.2 cm (24.1 to 25.5) |
| Range | 0.313 m (0.261 to 0.389) | 0.347 m (0.301 to 0.372) |
| Object points | 1058 (585 to 1525) | 557 (288 to 1051) |
| Depth drift in the dwell, p5-p95 | 4.6 cm (1.5 to 10.2) | 5.6 cm (3.2 to 9.2) |
| Range noise at the object | 1.9 mm (1.5 to 2.8) | 5.9 mm (4.8 to 6.2), includes curvature and aim wobble |

* Session 121018/121022/121055 used to fail in the cleaner. Since cleaner `45db92e` it is accepted in `drift_slab` mode, at
  14.0 x 22.5 cm. It is reported as measured: the uncleaned patch is also narrow.
* Bottle width is inflated by zone blur and wobble. Photos suggest about 7 cm; that is an estimate, not a measurement.

**Not measured:** the back and true depth of either object, absolute sensor position (translation is untracked), object
dimensions with a ruler, room height (ceiling z is above the resting device), a real clock (seconds assume 15 Hz).

## 4. Simulated data

`scripts/gen_final.py`: simulate raw file -> `thin` -> `clean(origin)` -> `to_canonical` -> `sample_2048`, target = complete
surface in the same transform. Scans the cleaner rejects are skipped and counted.

| Regime | What it is |
|---|---|
| `real`, `majproj/sim_real.py` | measured frames from `configs/real_bank.npz`, real object cut out, new object ray cast through the VL53L5CX model at the session's measured place and drift |
| `robust`, `majproj/sim_raw.py` | invented rooms, wider ranges |

| Split | Attempted | Kept | Kept `real` / `robust` |
|---|---|---|---|
| train | 71,680 | 64,847; 61,137 after dropping scans with under half the cleaned points on the object | 59,563 / 5,284 |
| val | not quoted | 3,675 | 3,344 / 331 |
| test | 2,048 | 1,857 | 1,711 / 146 |

On test the cleaner keeps 97.9 % of `real` regime scans (1711 of 1748) and 48.7 % of `robust` ones (146 of 300). Dataset hash
`1fa55370`, generated at code `45db92e`. Scans are drawn by kind with weights 3 cuboid, 1 cube, 1 slab, 1 tower, 3 bottle, 2 wavy_bottle,
1 can, 1 jar, 1 rod, 1 pipe.

Object sizes in the `real` regime (metres, `sample_shape`):

| Kind | Size |
|---|---|
| cuboid | each side 0.03 to 0.6, log-uniform |
| cube | side 0.03 to 0.6, log-uniform |
| slab | thickness 0.015 to 0.05, sides 0.12 to 0.6, log-uniform; thin along width (edge on) 25 %, depth (facing) 60 %, height (lying flat) 15 % |
| tower | base 0.03 to 0.15, height 0.3 to 0.6 |
| bottle, wavy_bottle | radius 0.02 to 0.06, height 0.12 to 0.40, shoulder and neck; wavy grips on the second |
| can | radius 0.025 to 0.06, height 0.08 to 0.2 |
| jar | radius 0.035 to 0.08, height 0.07 to 0.2 |
| rod | radius 0.01 to 0.025, height 0.2 to 0.6 |
| pipe | radius 0.015 to 0.08, height 0.15 to 0.6, wall 2 to 10 mm |

**Assumed, not measured:** lateral and vertical drift, object reflectivity and gloss, what lies behind the old object (pooled 5 deg
direction cells). **Fitted to the real sessions:** face range offset, object dropout, noise and validity tables. Everything in
the `real` regime is one room, one table, one device. Objects are upright. The complete target is the true surface at the mean
sensor pose, not a measured surface.

## 5. How real sessions are used

| Use | 18 sessions | 121600 (box), 123815 (bottle) |
|---|---|---|
| Cleaner design and thresholds | yes | yes |
| Simulator calibration (fitted constants) | yes | yes: the fit saw all sessions |
| Room bank for the `real` regime | yes, object cut out | **no** |
| Real training stage, with full reference shapes registered to each scan | yes | **no** |
| Checkpoint selection | no | **no** |

A real result counts as held out only for those two sessions. Full per-session folds: not done.

## 6. Privacy and licensing

No personal data. Sensor geometry follows the team's MIT-licensed viewer (VL53L5CX-BNO08X-viewer, (c) 2026 Henrique Ferrolho) and
ST's per-zone angle tables. AdaPoinTr-PCN weights come from the PoinTr project (MIT).
