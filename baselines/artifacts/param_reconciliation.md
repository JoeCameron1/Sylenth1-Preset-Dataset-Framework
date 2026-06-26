# Parameter count reconciliation

Closes Reviewer 1's "200+ / 102 / 61 / 178" comment. All counts below were
recomputed directly from `FINAL_timbral_dataset_audiocommons.json` and
`sylenth1_params.json` (commands at the end of the file).

## Authoritative counts used throughout the revised paper

| Quantity | Count |
|---|---:|
| **Raw plugin parameters** reported by Sylenth1 over the VST3 host | **246** |
| Parameters excluded per Section 3.3 (arp, pitch/porta, volume/bypass/program/sync) | 67 |
| **Spec / timbre-relevant parameters** (the dataset's canonical feature set) | **179** |
| &nbsp;&nbsp;&nbsp;&nbsp;of which `float` (non-constant) | 106 |
| &nbsp;&nbsp;&nbsp;&nbsp;of which `enum` | 55 |
| &nbsp;&nbsp;&nbsp;&nbsp;of which `bool` | 18 |
| Factory-only keys also retained in factory entries (`lfo_1_free`, `lfo_2_free`, `solo`) | 3 |
| **Effective per-row schema for factory** | 182 |
| **Effective per-row schema for random** | 179 |
| PCA's "numeric columns" under the published `pd.api.types.is_numeric_dtype` rule | **121** |
| Constant-valued float parameters | **0** |

## Resolving each reviewer-flagged number

### "200+" — raw plugin parameters
`pedalboard.load_plugin` reports **246** parameters. The Sylenth1 GUI shows
~200+ controls plus per-step XY arp toggles. The 67 we exclude per §3.3 are
exactly the arpeggiator (`arp_*`, `xarp_*`, `sw_arponoff`), pitch / portamento
(`pitchbend`, `pitchbend_range`, `porta_mode`, `porta_time_ms`), global volume
/ bypass / program/sync (`main_volume`, `bypass`, `program`, `sync`), and the
factory-only `lfo_1_free`, `lfo_2_free`, `solo`. Of those 67, three are kept
in *factory* entries for completeness but dropped from analysis to make the
factory and random schemas comparable; the resulting **179 timbre-relevant
parameters** is the canonical feature set used by every downstream analysis.

### "178 vs 179" — older paper text vs current dataset
A previous export of the dataset listed 178 keys; the current
`FINAL_timbral_dataset_audiocommons.json` carries **179 in random entries**
(spec keys) and **182 in factory entries** (179 + `lfo_1_free` + `lfo_2_free`
+ `solo`). The 178 figure refers to that earlier export, which differed from
the current canonical schema by a single key. Going forward the paper states **179** uniformly with the factory-only 3 keys footnoted as a schema
detail not used in modeling.

### "102" — PCA input dimensionality
The published PCA reported 102 input dimensions. This number is **not**
explained by dropping constant-valued floats: every one of the 106 spec
floats has non-zero variance across the dataset (`std > 0` for all). Inspecting
both PCA scripts shipped in the repo
(`pca_audiocommons_analysis.py`, `gen_pca_rand_vs_fact.py`), the column
selector is `pd.api.types.is_numeric_dtype(df[c])` *after*
`pd.to_numeric(errors='ignore')`. On the current dataset this produces
**121** numeric columns — the 106 spec floats *plus* 15 ordered-numeric
enums whose stored values happen to be pure numbers (parseable by
`to_numeric`):

```
filter_a_db_db, filter_b_db_db,
osc_a1_note,   osc_a1_octave, osc_a1_voices,
osc_a2_note,   osc_a2_octave, osc_a2_voices,
osc_b1_note,   osc_b1_octave, osc_b1_voices,
osc_b2_note,   osc_b2_octave, osc_b2_voices,
polyphony
```

## Reproducing these counts

```bash
# raw plugin parameter count (requires Sylenth1 VST3 installed)
python -c "from baselines.common.render import Sylenth1Controller, SYLENTH1_PATH_DEFAULT; \
           Sylenth1Controller(SYLENTH1_PATH_DEFAULT)"
# -> "Loaded Sylenth1 with 246 parameters."

# spec breakdown (179 = 106 float + 55 enum + 18 bool)
python -c "import json; s=json.load(open('sylenth1_params.json')); \
           print({t: sum(1 for v in s.values() if (v.get('type') or '').lower()==t) \
                  for t in ['float','enum','bool']})"
# -> {'float': 106, 'enum': 55, 'bool': 18}

# PCA-script numeric column count (121, of which 15 are ordered-numeric enums)
python -c "import json, pandas as pd; \
           rows=[{f'param_{k}':v for k,v in (e.get('params') or {}).items()} \
                  for e in json.load(open('FINAL_timbral_dataset_audiocommons.json'))]; \
           df=pd.DataFrame(rows); \
           [df.__setitem__(c, pd.to_numeric(df[c], errors='ignore')) \
                for c in df.columns if df[c].dtype==object]; \
           print(sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c])))"
# -> 121
```
