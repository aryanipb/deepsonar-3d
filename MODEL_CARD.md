# Model card: AdaPoinTr fine-tuned for one-view ToF scans

**Status, 2026-11-02.** The final model is run `final7`, weights selected at step 138,000. It is the model in the web app, APK and AppImage, and it passes **10 of 18** gates fixed before any result was seen ([`docs/evidence/evaluation.md`](docs/evidence/evaluation.md)).

## Summary

| | |
|---|---|
| Architecture | AdaPoinTr (Yu et al., *IEEE TPAMI 2023*; PoinTr, ICCV 2021), official code (MIT), PCN configuration, `majproj/model/` |
| Starting weights | Authors' AdaPoinTr-PCN checkpoint: 8 classes, neither box nor bottle |
| Final checkpoint | `adapointr_final.pth` (release asset), EMA weights, step 138,000, sha256 `ddfaa654b62dd0e294eba887e62a9e558a61a55ed642b1b1b5d0f85083448895` |
| Input | 2048 × 3 float32 in the canonical frame (`majproj/canon.py`). Geometry only: no label, kind, size or template reaches the network |
| Output | 16,384 × 3, the fine output `model(x)[1]`, mapped back to metres |
| ONNX | `webapp/model/`, opset 17, 139 MB, sha256 `1f3accbf67b5d60fdd01a4defa97925a4d63c54449e7e3ddf716109244ceb944`. Passes all 23 probes, 22 of them to below 1e-5 Chamfer ([log](docs/evidence/onnx_parity_final7.log)) |
| Dataset hash | `1fa55370a4327799cf85f775c736fe86c476f42b90b77104dd45cf9286450227` |

## Intended use

Show that a completion network trained on device-faithful simulation and a small set of real sessions can turn one aimed scan into a plausible complete object of the trained families: cuboids and cylinders.

Not for measuring depth or the back of an object, for unseen rooms or devices without new validation, for objects outside the trained families, or for objects lying flat.

## Training

| | |
|---|---|
| Compute | One NVIDIA H100, 15.8 hours across seven runs |
| Steps | 142,129; selected at 138,000 by validation select score 16.755 |
| Data | 61,137 simulated training scans; 18 real sessions |
| Validation, test | 3,675 and 1,857 simulated scans |
| Batch | 16 simulated, plus 8 real from stage 2 at loss weight 3 |
| Learning rate | 1e-4, later 5e-5; weight decay 5e-4; 500 warm-up steps; EMA |
| Precision | float16 autocast, gradient clip 10 |
| Loss | Sparse and dense Chamfer L1, AdaPoinTr denoising term, real-scan term |

**Real-scan targets.** Most real sessions train against a parametric model of the known object, registered to each scan's front. This teaches the network these two objects, so the real-scan result is instance-specific. The box model uses an assumed 2 cm depth; neither object was measured with a ruler.

**Hold-out.** Sessions `121600` (box) and `123815` (bottle) were fixed before any result and never enter training or the simulator's room bank. Simulated validation picked the weights inside each run, but choices between training branches also looked at these two sessions, and the cleaner's thresholds and the simulator's fitted constants saw all sessions. They are not a pure test set.

**Augmentation.** Partial and target together: mirror, yaw, scale. Partial only: drop, half-plane crop, local hole, depth copies, jitter. The canonical frame is recomputed from the augmented partial.

## Evaluation

All figures and all 18 gates: [`docs/evidence/evaluation.md`](docs/evidence/evaluation.md). Per-session history: [`docs/metrics/`](docs/metrics/).

| Measure | Result |
|---|---|
| Held-out box vs reference shape | 3.1 mm |
| Held-out bottle vs reference shape | 4.8 mm |
| Simulated median CD, room-faithful regime | 3.7 mm (1,711 scans) |
| Simulated median CD, invented rooms | 8.1 mm (146 scans) |
| Box width and height spread (std/mean) | 0.018 and 0.010 over 11 sessions |
| Bottle width and height spread (std/mean) | 0.099 and 0.024 over 9 sessions |

## Known limits

- The depth of a box can't be measured from one face. The output depth is a prediction, and the real box's output varies from 5.2 to 9.1 cm across sessions.
- Both probe gates fail: margin 0.92 against 3.0, and a width ratio of 0.882 against 0.75. Outputs are pulled toward learned objects, and generalisation is unproven.
- Sessions `123028` and `123523` fail the bottle roundness gates.
- The cleaner accepts upright targets only. A wrong patch gives a confident wrong shape.
- Quality falls sharply beyond 0.40 m and in rooms unlike the training room.
- One room and one device; two sessions held out; no full leave-one-out.

## Licence

Model code: MIT (`majproj/model/LICENSE_PoinTr`). Everything else: see [`LICENSE`](LICENSE).
