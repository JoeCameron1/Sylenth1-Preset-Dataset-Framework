# Multi-pitch sensitivity — notes

Companion to `multipitch_sensitivity.csv` and `multipitch_renders.csv`
(Task 7, spec §8).

## Setup

200 presets stratified by brightness quintiles (40 per quintile), re-rendered
at MIDI 45, 57, 69, 81, 93 (A2 through A6) through the IDENTICAL normalisation
chain used for the A4 dataset. 1000 plugin renders, ~18 min serial.

## Mean |deviation from A4| per (descriptor, pitch), 0-100 units

| descriptor | A2 | A3 | A4 | A5 | A6 |
|---|---:|---:|---:|---:|---:|
| brightness |  9.46 |  5.78 | 0.00 |  6.06 | 11.71 |
| depth      | 16.44 | 10.73 | 0.00 |  9.96 | 16.51 |
| hardness   |  8.13 |  5.96 | 0.00 |  6.07 |  8.79 |
| roughness  | 11.19 |  7.70 | 0.00 |  6.97 | 10.18 |
| warmth     |  9.97 |  6.84 | 0.00 |  7.13 | 11.28 |
| sharpness  |  8.93 |  6.25 | 0.00 |  7.13 | 13.63 |
| boominess  | 14.14 | 10.36 | 0.00 | 10.42 | 16.10 |

## What this says

Descriptor values **are pitch-dependent**, and the shift is substantial at
the extremes:

* **Depth and boominess shift the most** (~16 units at A2 and A6). On a 0-100
  AudioCommons scale this is ~16% of the full range; descriptor labels at A4
  do not transfer directly to either octave away.
* **Hardness is the most pitch-invariant** (~8 units at extremes). Its
  computation appears more robust to register changes.
* Shifts are roughly symmetric around A4 (each descriptor's A2 and A6
  deviations are within a few units of each other).
* A3 and A5 are intermediate (5-11 units). Adjacent octaves are not free.

## Implication for the baselines / paper

This **measures** the pitch-dependence limitation rather than leaving it
unaddressed — exactly what Reviewer 1 ("run more than one pitch") and
Reviewer 3 (Reymore et al. citation) asked for. The A4-only dataset and
baselines remain a coherent benchmark *at A4*, with this study as the
justification — but the revised paper should clearly state:

* The A4 descriptors do **not** transfer linearly to other pitches; a future
  multipitch extension is needed for register-spanning timbral retrieval /
  inversion claims.
* The CVAE round-trip eval (Task 5) is bounded to the A4 condition under
  which the dataset was built; reporting it as "1-octave performance" rather
  than "pitch-invariant performance" is the honest framing.
* Depth and boominess are the most pitch-sensitive descriptors — when the
  paper discusses these in Section 5, it should note that their values are
  meaningful relative to register rather than absolute.

The heatmap (`baselines/artifacts/figures/multipitch_heatmap.png`) is the
single figure that conveys all of the above.
