# Pathology audit set

44 WAVs curated for a listening pass, four per criterion (see
`baselines/artifacts/results/pathology_candidates.csv` for the row-level
metadata: preset id, name, kind, reason, the 7 AC descriptor values).

Filename convention: `{kind}_{reason}_{preset_id}.wav`.

## Criteria (4 presets each)

* `convergent_outlier:<descr>~<feature>` — presets where AC descriptor
  disagrees most strongly (in z-score space) with a simple acoustic
  predictor. If the audio "sounds bright" but AC `brightness` is low (or
  vice-versa), the AC label is the suspect. Pairs covered:
  brightness~centroid, depth~low_band_ratio, boominess~low_band_ratio,
  warmth~high_band_ratio.
* `contradictory_pair:<a>+<b>` — presets with both descriptors high when
  they should typically trade off: brightness+boominess, depth+sharpness,
  warmth+sharpness.
* `roughness:small_nonzero` — smallest *positive* roughness values
  (just above the 0-floor). If audio actually sounds smooth, these are
  the false-positives at the threshold boundary.
* `roughness:very_high` — top of the roughness distribution. Do they
  actually sound rough?
* `extreme_zscore` — any AC descriptor more than 3 stddev from the
  per-descriptor mean. Statistical anomalies.
* `max_distance_from_factory` — random presets with the largest assigned
  distance from their nearest factory progenitor (per `splits.json`).
  R1 flagged these as the "interpolation tail" that may include
  unmusical patches.

## Suggested listening protocol

For each WAV, listen and decide:

1. Does the AC label seem perceptually correct?
2. If `roughness:small_nonzero`: does it sound rough at all, or smooth?
3. If `contradictory_pair`: do both qualities really co-occur?
4. If `max_distance_from_factory`: does the patch sound musically usable,
   or does it sound like a "preset gone wrong"?

A short report of agreement/disagreement counts per criterion supplies the
qualitative response to R2's revision request.
