# Completion evaluation: FINAL

Checkpoint adapointr_final.pth, step 138000, sha256 ddfaa654b62dd0e2. Label: FINAL. 10 of 18 gates pass, 8 fail.
Simulation: test split, all 1857 scans, 9.3 % of drawn test scans were rejected by the cleaner and are not in these numbers. Val picked the checkpoint and is not used here.
Real: 20 sessions, each once, 2 are leave-session-out at every stage (123815 121600, starred in the gates). Sessions in the real stage of this checkpoint: 18 sessions. The other 18 fed the simulator's room bank, so they are not held out and are never labelled so. The cleaner's thresholds were tuned with all sessions in view, so no session is held out from the cleaner.
The back and depth of a real object were never measured. Real depth is what the network claims; the only check on it is agreement across sessions and with the simulation error on depth.
The network input is the 2048 canonical points only. No label, kind or size reaches it at inference. All gate thresholds were written down before the first result.

## Gates

| gate | result | detail |
|---|---|---|
| sim: median CD below returning the input, every kind | PASS | failing kinds: none |
| sim: median CD below official PCN weights, every kind | PASS | failing kinds: none |
| sim: median CD at most 5.0 mm, every kind | FAIL | failing kinds: cuboid, cube, slab, tower |
| sim: mean F@10 mm at least 0.8, every kind | FAIL | failing kinds: cuboid, cube |
| real: every one of the 20 sessions gives an output | PASS | rejected: none |
| real: one-sided median at most 10.0 mm | FAIL | 2 sessions fail: 122622 123146 |
| bottle: section residual / radius at most 0.1 | FAIL | 2 sessions fail: 123028 123523 |
| bottle: angular coverage at least 0.9 | FAIL | 2 sessions fail: 123028 123523 |
| bottle: section aspect at least 0.8 | FAIL | 2 sessions fail: 123028 123523 |
| box: median face flatness at most 5.0 mm | PASS | all sessions pass |
| box: worst right-angle error at most 5.0 deg | PASS | all sessions pass |
| box: at least 0.8 of the output within 8 mm of a fitted face | PASS | all sessions pass |
| box: width std/mean across sessions at most 0.1 | PASS | 0.018 over 11 sessions |
| box: height std/mean across sessions at most 0.1 | PASS | 0.010 over 11 sessions |
| bottle: width std/mean across sessions at most 0.1 | PASS | 0.099 over 9 sessions |
| bottle: height std/mean across sessions at most 0.1 | PASS | 0.024 over 9 sessions |
| probes: plate, rod, cube, noise differ, margin at least 3.0 | FAIL | margin 0.92 |
| probe: width/height of the output follows a halved width, ratio at most 0.75 | FAIL | ratio 0.882; 0.5 is full tracking, 1 is a fixed shape |

## Simulation by kind, cleaner on the object, regime real (1637 scans)

CD-L1 is the mean of both directions, in millimetres on the scan's own scale. Official PCN weights were trained in PCN's normalisation, not this canonical frame; they get the same inputs as asked. Unseen recall: target points over 1 cm from every input point that an output point reaches within 10 mm.

| group | n | CD mean mm | CD median mm | official PCN median | input median | F@5 mm | F@10 mm | official F@10 | input F@10 | unseen recall@10 mm | input unseen recall | W err mm | H err mm | D err mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuboid | 296 | 7.8 | 6.5 | 30.9 | 31.7 | 0.540 | 0.782 | 0.312 | 0.303 | 0.651 | 0.000 | 19.3 | 4.7 | 62.4 |
| cube | 97 | 6.9 | 5.7 | 50.9 | 47.3 | 0.541 | 0.789 | 0.230 | 0.270 | 0.685 | 0.000 | 21.9 | 3.7 | 33.6 |
| slab | 77 | 7.3 | 6.8 | 14.4 | 14.1 | 0.558 | 0.830 | 0.452 | 0.446 | 0.723 | 0.000 | 15.6 | 4.2 | 70.7 |
| tower | 102 | 5.4 | 5.3 | 15.2 | 17.8 | 0.675 | 0.907 | 0.440 | 0.402 | 0.837 | 0.000 | 6.3 | 6.5 | 30.9 |
| bottle | 371 | 3.0 | 2.9 | 13.6 | 15.8 | 0.880 | 0.991 | 0.464 | 0.440 | 0.996 | 0.000 | 3.2 | 5.1 | 3.3 |
| wavy_bottle | 220 | 3.0 | 2.9 | 13.4 | 15.5 | 0.875 | 0.990 | 0.475 | 0.444 | 0.996 | 0.000 | 3.8 | 4.1 | 3.7 |
| can | 124 | 2.9 | 2.7 | 17.4 | 18.4 | 0.894 | 0.985 | 0.422 | 0.410 | 0.991 | 0.000 | 4.4 | 3.7 | 4.6 |
| jar | 129 | 3.2 | 3.1 | 22.4 | 22.8 | 0.854 | 0.979 | 0.365 | 0.368 | 0.983 | 0.000 | 5.2 | 3.1 | 5.2 |
| rod | 100 | 3.4 | 3.2 | 9.1 | 11.4 | 0.868 | 0.984 | 0.579 | 0.516 | 0.994 | 0.000 | 4.3 | 9.7 | 6.2 |
| pipe | 121 | 3.9 | 3.8 | 19.1 | 22.0 | 0.783 | 0.961 | 0.444 | 0.427 | 0.947 | 0.000 | 5.0 | 8.4 | 4.9 |

## Simulation by kind, every accepted scan (1857 scans)

Includes scans where the cleaner kept the wrong object: no network can fix those, and mean far above median shows them.

| group | n | CD mean mm | CD median mm | official PCN median | input median | F@5 mm | F@10 mm | official F@10 | input F@10 | unseen recall@10 mm | input unseen recall | W err mm | H err mm | D err mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cuboid | 371 | 51.1 | 7.3 | 36.4 | 36.3 | 0.465 | 0.678 | 0.273 | 0.261 | 0.572 | 0.000 | 25.4 | 6.1 | 67.9 |
| cube | 125 | 78.9 | 6.4 | 51.9 | 52.7 | 0.475 | 0.690 | 0.216 | 0.245 | 0.613 | 0.000 | 23.3 | 4.9 | 36.0 |
| slab | 93 | 47.0 | 7.2 | 16.1 | 15.5 | 0.484 | 0.726 | 0.406 | 0.390 | 0.634 | 0.000 | 21.3 | 4.7 | 74.5 |
| tower | 115 | 8.2 | 5.3 | 16.4 | 18.4 | 0.627 | 0.863 | 0.423 | 0.381 | 0.811 | 0.000 | 6.6 | 8.0 | 31.6 |
| bottle | 400 | 4.9 | 2.9 | 14.0 | 16.2 | 0.853 | 0.973 | 0.452 | 0.426 | 0.983 | 0.000 | 3.3 | 5.5 | 3.4 |
| wavy_bottle | 244 | 10.7 | 3.0 | 13.8 | 16.4 | 0.829 | 0.955 | 0.455 | 0.424 | 0.968 | 0.000 | 4.1 | 4.7 | 4.1 |
| can | 131 | 6.1 | 2.8 | 17.4 | 18.6 | 0.873 | 0.970 | 0.418 | 0.403 | 0.980 | 0.000 | 4.6 | 3.8 | 4.8 |
| jar | 139 | 3.6 | 3.1 | 22.8 | 23.3 | 0.832 | 0.968 | 0.361 | 0.357 | 0.972 | 0.000 | 5.3 | 3.3 | 5.4 |
| rod | 107 | 7.7 | 3.3 | 9.2 | 11.5 | 0.838 | 0.960 | 0.561 | 0.499 | 0.964 | 0.000 | 4.4 | 10.5 | 6.9 |
| pipe | 132 | 8.8 | 3.9 | 19.6 | 23.2 | 0.750 | 0.931 | 0.428 | 0.407 | 0.920 | 0.000 | 5.4 | 9.0 | 5.2 |

## Simulation by distance, every accepted scan

| group | n | CD mean mm | CD median mm | official PCN median | input median | F@5 mm | F@10 mm | official F@10 | input F@10 | unseen recall@10 mm | input unseen recall | W err mm | H err mm | D err mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00-0.28 m | 288 | 16.8 | 3.9 | 17.2 | 19.3 | 0.721 | 0.886 | 0.414 | 0.384 | 0.855 | 0.000 | 6.1 | 5.9 | 6.9 |
| 0.28-0.33 m | 942 | 19.8 | 3.8 | 17.0 | 19.0 | 0.714 | 0.883 | 0.401 | 0.379 | 0.848 | 0.000 | 5.8 | 5.3 | 7.4 |
| 0.33-0.40 m | 543 | 23.5 | 3.7 | 17.1 | 18.3 | 0.739 | 0.882 | 0.384 | 0.378 | 0.852 | 0.000 | 5.5 | 4.8 | 7.5 |
| 0.40 m and over | 84 | 68.1 | 9.4 | 26.1 | 29.5 | 0.350 | 0.571 | 0.269 | 0.212 | 0.604 | 0.000 | 14.6 | 27.7 | 23.4 |

## Simulation by regime, every accepted scan

| group | n | CD mean mm | CD median mm | official PCN median | input median | F@5 mm | F@10 mm | official F@10 | input F@10 | unseen recall@10 mm | input unseen recall | W err mm | H err mm | D err mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 1711 | 20.7 | 3.7 | 16.9 | 18.7 | 0.732 | 0.889 | 0.400 | 0.383 | 0.854 | 0.000 | 5.7 | 5.1 | 7.1 |
| robust | 146 | 45.4 | 8.1 | 23.4 | 27.8 | 0.399 | 0.632 | 0.299 | 0.239 | 0.665 | 0.000 | 13.9 | 26.5 | 23.0 |

## Same shape checks on held-out simulated outputs (median per kind)

| kind | n | section diameter cm | section residual mm | residual / radius | angular coverage | section aspect | face flatness mm | worst right-angle error deg | share within 8 mm of face |
|---|---|---|---|---|---|---|---|---|---|
| cuboid | 371 | 24.7 | 15.8 | 0.127 | 0.44 | 0.53 | 7.0 | 5.1 | 0.96 |
| cube | 125 | 23.8 | 14.7 | 0.115 | 0.67 | 0.63 | 7.2 | 3.5 | 0.97 |
| slab | 93 | 25.6 | 17.8 | 0.133 | 0.44 | 0.37 | 8.4 | 6.5 | 0.95 |
| tower | 115 | 15.4 | 13.4 | 0.162 | 0.58 | 0.65 | 6.2 | 6.6 | 0.97 |
| bottle | 400 | 7.8 | 1.2 | 0.034 | 1.00 | 0.95 | 5.7 | 47.2 | 0.87 |
| wavy_bottle | 244 | 7.6 | 1.3 | 0.036 | 1.00 | 0.95 | 5.6 | 44.4 | 0.89 |
| can | 131 | 9.0 | 1.6 | 0.036 | 1.00 | 0.95 | 3.4 | 13.0 | 0.97 |
| jar | 139 | 11.5 | 1.6 | 0.029 | 1.00 | 0.95 | 4.3 | 13.1 | 0.95 |
| rod | 107 | 5.3 | 6.5 | 0.130 | 0.85 | 0.79 | 3.0 | 6.2 | 1.00 |
| pipe | 132 | 10.1 | 2.7 | 0.055 | 1.00 | 0.93 | 4.9 | 7.2 | 0.93 |

## Real sessions, FINAL: size and one-sided fit

One-sided fit: every cleaned point to its nearest output point. It says the output covers what was measured, nothing about the back.

| object | session | leave-session-out | in real stage | cleaner | cleaned points | seen face long side cm | output width cm | output height cm | output depth cm (never measured) | one-sided median mm | one-sided p95 mm | official PCN one-sided median mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BOTTLE | 122622 | no | yes | ok | 288 | 24.8 | 7.1 | 22.8 | 7.1 | 10.3 | 27.0 | 0.3 |
| BOTTLE | 122735 | no | yes | ok | 523 | 25.2 | 7.2 | 23.0 | 7.2 | 9.6 | 26.6 | 1.0 |
| BOTTLE | 122907 | no | yes | ok | 641 | 24.2 | 6.9 | 22.1 | 6.9 | 8.8 | 23.2 | 1.6 |
| BOTTLE | 123028 | no | yes | ok | 557 | 25.5 | 6.8 | 24.3 | 9.1 | 0.2 | 1.1 | 1.1 |
| BOTTLE | 123146 | no | yes | ok | 345 | 24.7 | 7.2 | 23.1 | 7.2 | 10.5 | 26.3 | 0.4 |
| BOTTLE | 123259 | no | yes | ok | 1051 | 25.0 | 7.1 | 22.9 | 7.1 | 9.7 | 25.7 | 2.4 |
| BOTTLE | 123523 | no | yes | ok | 410 | 24.9 | 5.0 | 23.0 | 7.8 | 0.2 | 0.6 | 0.5 |
| BOTTLE | 123658 | no | yes | ok | 1040 | 25.5 | 7.3 | 23.3 | 7.3 | 8.9 | 26.9 | 2.4 |
| BOTTLE | 123815 | yes | no | ok | 680 | 25.4 | 6.9 | 22.6 | 6.9 | 8.5 | 22.3 | 1.7 |
| BOX | 114126 | no | yes | ok | 1445 | 25.5 | 18.5 | 23.5 | 6.3 | 5.3 | 15.8 | 4.6 |
| BOX | 115102 | no | yes | ok | 1253 | 21.6 | 18.6 | 23.4 | 5.2 | 6.0 | 19.8 | 4.5 |
| BOX | 115330 | no | yes | ok | 1085 | 24.0 | 18.3 | 23.4 | 6.9 | 4.7 | 12.6 | 3.9 |
| BOX | 115840 | no | yes | ok | 1058 | 23.0 | 18.7 | 23.5 | 5.2 | 4.7 | 10.0 | 3.7 |
| BOX | 120051 | no | yes | ok | 1045 | 23.9 | 18.4 | 23.5 | 6.4 | 5.0 | 10.2 | 3.4 |
| BOX | 120408 | no | yes | ok | 1525 | 25.3 | 18.3 | 23.4 | 6.9 | 5.9 | 21.8 | 3.9 |
| BOX | 120745 | no | yes | ok | 906 | 20.9 | 18.1 | 23.4 | 7.3 | 5.7 | 16.8 | 4.4 |
| BOX | 121055 | no | yes | ok | 585 | 22.5 | 18.2 | 23.5 | 7.7 | 4.2 | 9.1 | 1.3 |
| BOX | 121223 | no | yes | ok | 1078 | 23.0 | 17.6 | 23.5 | 9.1 | 6.6 | 24.9 | 3.5 |
| BOX | 121457 | no | yes | ok | 1042 | 22.3 | 18.3 | 23.5 | 7.0 | 5.5 | 15.5 | 3.6 |
| BOX | 121600 | yes | no | ok | 971 | 22.9 | 18.9 | 24.3 | 6.8 | 5.7 | 16.0 | 3.7 |

## Real sessions, FINAL: shape of the output

Both checks run on every session with no label, so a box row under the round columns and a bottle row under the box columns show how far apart the two shapes are.

| object | session | leave-session-out | section diameter cm | section residual mm | residual / radius | angular coverage | section aspect | faces | face flatness mm | worst face mm | worst right-angle error deg | share within 8 mm of face | worst face fill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BOTTLE | 122622 | no | 6.7 | 0.5 | 0.015 | 1.00 | 0.96 | 6 | 4.5 | 6.3 | 2.5 | 0.94 | 0.44 |
| BOTTLE | 122735 | no | 6.7 | 0.5 | 0.015 | 1.00 | 0.96 | 6 | 4.5 | 6.5 | 3.5 | 0.94 | 0.44 |
| BOTTLE | 122907 | no | 6.5 | 0.5 | 0.014 | 1.00 | 0.95 | 6 | 4.4 | 6.3 | 2.7 | 0.95 | 0.47 |
| BOTTLE | 123028 | no | 4.8 | 10.9 | 0.456 | 0.81 | 0.45 | 6 | 6.0 | 6.5 | 49.5 | 0.80 | 0.22 |
| BOTTLE | 123146 | no | 6.7 | 0.5 | 0.015 | 1.00 | 0.94 | 6 | 4.5 | 6.4 | 2.9 | 0.94 | 0.50 |
| BOTTLE | 123259 | no | 6.7 | 0.5 | 0.016 | 1.00 | 0.95 | 6 | 4.6 | 6.4 | 3.0 | 0.94 | 0.47 |
| BOTTLE | 123523 | no | 4.0 | 8.2 | 0.409 | 0.75 | 0.53 | 6 | 4.9 | 6.0 | 71.3 | 0.86 | 0.31 |
| BOTTLE | 123658 | no | 6.8 | 0.5 | 0.015 | 1.00 | 0.95 | 6 | 4.7 | 6.5 | 3.1 | 0.93 | 0.42 |
| BOTTLE | 123815 | yes | 6.5 | 0.7 | 0.021 | 1.00 | 0.93 | 6 | 4.3 | 6.2 | 3.4 | 0.95 | 0.50 |
| BOX | 114126 | no | 11.4 | 28.1 | 0.495 | 0.98 | 0.17 | 6 | 0.9 | 1.1 | 2.1 | 1.00 | 1.00 |
| BOX | 115102 | no | 11.4 | 28.2 | 0.493 | 0.96 | 0.17 | 6 | 0.9 | 1.1 | 1.4 | 1.00 | 1.00 |
| BOX | 115330 | no | 11.4 | 28.1 | 0.492 | 0.98 | 0.18 | 6 | 0.8 | 1.0 | 2.7 | 1.00 | 1.00 |
| BOX | 115840 | no | 11.5 | 28.2 | 0.492 | 0.96 | 0.18 | 6 | 0.8 | 1.1 | 3.0 | 1.00 | 1.00 |
| BOX | 120051 | no | 11.4 | 28.0 | 0.492 | 0.94 | 0.18 | 6 | 0.9 | 1.1 | 3.2 | 1.00 | 1.00 |
| BOX | 120408 | no | 11.4 | 28.1 | 0.492 | 0.92 | 0.17 | 6 | 1.0 | 1.2 | 2.5 | 1.00 | 1.00 |
| BOX | 120745 | no | 11.3 | 28.0 | 0.494 | 0.96 | 0.17 | 6 | 0.9 | 1.1 | 4.6 | 1.00 | 1.00 |
| BOX | 121055 | no | 11.5 | 28.2 | 0.491 | 0.94 | 0.18 | 6 | 0.8 | 1.1 | 0.9 | 1.00 | 1.00 |
| BOX | 121223 | no | 11.4 | 28.0 | 0.491 | 0.94 | 0.18 | 6 | 1.0 | 1.2 | 2.6 | 1.00 | 0.98 |
| BOX | 121457 | no | 11.3 | 28.1 | 0.496 | 0.96 | 0.17 | 6 | 0.8 | 1.0 | 2.9 | 1.00 | 1.00 |
| BOX | 121600 | yes | 11.7 | 29.0 | 0.497 | 1.00 | 0.17 | 6 | 0.9 | 1.1 | 2.3 | 1.00 | 1.00 |

## Consistency across sessions of the one box and the one bottle

Pairwise distance is between two sessions' outputs, centred, at the best rotation about the vertical. Box against bottle for scale: 19.4 mm.

| object | sessions | width cm mean / std / range | height cm | depth cm (never measured) | width std/mean | height std/mean | depth std/mean | pairwise distance median mm | pairwise distance max mm |
|---|---|---|---|---|---|---|---|---|---|
| BOX | 11 | 18.4 / 0.3 / 17.6-18.9 | 23.5 / 0.2 / 23.4-24.3 | 6.8 / 1.0 / 5.2-9.1 | 0.018 | 0.010 | 0.153 | 3.2 | 5.1 |
| BOTTLE | 9 | 6.8 / 0.7 / 5.0-7.3 | 23.0 / 0.6 / 22.1-24.3 | 7.4 / 0.6 / 6.9-9.1 | 0.099 | 0.024 | 0.086 | 2.8 | 13.5 |

## Probes: output size and repeatability

Clouds are built in metres, not by the simulator: a 30 x 20 cm plate, a 2 cm rod 30 cm long, a 12 cm cube turned 35 degrees, uniform noise in a 20 cm cube, each with three noise seeds.

| probe | width cm | height cm | depth cm | worst same-probe distance (canonical x 1000) |
|---|---|---|---|---|
| plate | 30.9 | 20.0 | 10.0 | 10.7 |
| rod | 1.5 | 30.6 | 1.6 | 2.6 |
| cube | 11.1 | 11.9 | 12.5 | 10.9 |
| noise | 18.5 | 22.0 | 16.7 | 57.0 |

## Probes: distance between outputs (canonical x 1000)

Halved width probe on held-out box 121600: output width/height changes by a factor 0.882 (0.5 tracks fully, 1 means a fixed shape).

| pair | distance |
|---|---|
| plate/rod | 110.4 |
| plate/cube | 67.7 |
| plate/noise | 52.3 |
| rod/cube | 125.2 |
| rod/noise | 79.6 |
| cube/noise | 108.6 |
