# AC descriptor convergent validity on Sylenth1 A4 audio

Companion to `convergent_validity_correlations.csv`,
`convergent_validity_per_preset.csv`, and
`baselines/artifacts/figures/convergent_validity_heatmap.png`. Addresses
**Reviewer 1's main concern** that the AudioCommons models were developed and
validated primarily on real-world / musical-instrument audio, not synthesised
sounds: "the prediction quality in the specific setting must be evaluated...
at least for a subset the accuracy of the AudioCommons model's predictions on
the given audio (which is the Sylenth1 output) is measured and shown to be
sufficient."

## What we did

We **cannot** run a listening test from code, but we can run a *convergent-
validity* check: do the AC labels correlate with simple, well-understood
acoustic features computed directly from the audio? If AC `brightness`
correlates strongly with spectral centroid on Sylenth1 output, that is direct
evidence that the AC model is doing something sensible on this audio
distribution rather than producing arbitrary numbers.

Procedure: for every random preset's WAV (9,998 of the 10,000 with WAVs
present at run time, 0 silent / invalid), compute via librosa:

  * spectral centroid (Hz)
  * spectral rolloff at 85%
  * spectral bandwidth
  * spectral flatness (0..1)
  * fraction of spectral power in [0, 250) Hz (`low_band_ratio`)
  * fraction in [250, 2000) Hz (`mid_band_ratio`)
  * fraction in [4000, sr/2) Hz (`high_band_ratio`)
  * zero-crossing rate
  * RMS

Then compute Pearson and Spearman rank correlations between each AC
descriptor and each acoustic feature across the 9,998 presets.

## Headline correlations (Spearman r, full dataset)

| AC descriptor | strongest acoustic predictor | r | second predictor | r |
|---|---|---:|---|---:|
| brightness | spectral_centroid    | **+0.87** | high_band_ratio   | +0.85 |
| depth      | low_band_ratio       | **+0.88** | spectral_rolloff_85 | −0.76 |
| boominess  | low_band_ratio       | **+0.88** | spectral_rolloff_85 | −0.65 |
| warmth     | high_band_ratio      | **−0.78** | mid_band_ratio    | −0.39 |
| sharpness  | spectral_centroid    | **+0.77** | zero_crossing_rate | +0.76 |
| hardness   | zero_crossing_rate   | **+0.69** | spectral_flatness | +0.52 |
| roughness  | (no simple predictor strong; r < 0.4 across all)         |  |   |

Full matrix in `convergent_validity_correlations.csv`; visualised in the
companion heatmap (RdBu_r, symmetric).

## Interpretation

**5 of 7** AC descriptors have a Spearman correlation **above 0.7** with at
least one simple, intuitively-expected acoustic feature computed from the
Sylenth1 A4 audio:

  * brightness, depth, boominess: very strong (r ≥ 0.87 with their natural
    predictors) — the AC model's labels for these three descriptors track
    spectral energy distribution exactly as expected for synthetic audio.
  * sharpness and warmth: strong (r > 0.75); warmth correctly correlates
    *negatively* with high-band energy.
  * hardness: moderate (r = 0.69 with ZCR, 0.52 with flatness) — both
    intuitive predictors, but the labels carry information beyond either.
  * **roughness**: no simple acoustic feature predicts it strongly (max
    |r| < 0.4 across the feature set). Combined with the known 0-spike
    (≈9.8% of the dataset exactly 0), roughness is the descriptor whose
    AC-vs-perception agreement is least defensible from this analysis
    alone — the paper should flag it as the one requiring the most caution.

These results do not replace a perceptual study, but they directly answer
the "is the AC model even sensible on this audio?" challenge from R1 with
quantitative evidence on the full 10k random subset: **yes for 5 of 7
descriptors with high confidence, moderately for hardness, and not for
roughness from this evidence**.

## Companion: pathology candidate set

`pathology_candidates.csv` and `baselines/artifacts/pathology_audit/` (44
WAVs) provide a curated audit set selected by 11 algorithmic criteria
(convergent-validity outliers, contradictory descriptor pairs, roughness
edge cases, z-score outliers, presets maximally far from any factory
parent). This addresses Reviewer 2's "listening test on samples that
strongly deviate the most from the preset".
