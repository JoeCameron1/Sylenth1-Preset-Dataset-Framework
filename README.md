# Sylenth1-Preset-Dataset-Framework

## Dataset & Code Repository for the paper "A Dataset and Analysis Framework for Large-Scale Synthesizer Preset Generation"

This repository contains a dataset of presets from the Sylenth1 synthesizer along with the AudioCommons timbral model values for each preset. It also contains all the code used to create the dataset and for further analysis.

Below is an explanation of each file in this repository.

**Dataset Files:**
- ***Sylenth1_Full_Factory_Presets_AC_A4.json*** - This contains the 292 factory presets of Sylenth1 along with their corresponding AudioCommons timbral model values.
- ***Full_Random_Presets.json*** - This contains the 10,000 randomly generated presets from Sylenth1 along with their corresponding AudioCommons timbral model values.
- ***FINAL_timbral_dataset_audiocommons.json*** - This contains the total combined dataset of the 292 factory presets from 'Sylenth1_Full_Factory_Presets_AC_A4.json' and the 10,000 randomly generated presets from 'Full_Random_Presets.json' along with their corresponding AudioCommons timbral model values.
- ***random_preset_audio_snippets/*** - This folder contains the audio snippets for the 10,000 randomly generated presets used to calculate their corresponding AudioCommons timbral model values.

**Code Files:**
- ***sylenth1_preset_custom_app.py*** - This is the custom app used to interact with and control Sylenth1. Presets were created and saved to the dataset via this app. *Usage*: python sylenth1_preset_custom_app.py
- ***build_sylenth1_timbre_dashboard.py*** - This builds a Plotly dashboard for analysing the contents and distributions of the dataset's .JSON files. The figures from Section X in the paper are produced by this script. *Usage*:
- ***pca_audiocommons_analysis.py*** - This performs all the PCA analysis on the dataset's .JSON files. *Usage*:
- ***gen_ac_hist.py*** - Example. *Usage*:
- ***gen_pca_rand_vs_fact.py*** - Example. *Usage*:
- ***Sylenth1_Full_FactoryPresets_Timbre_Dashboard.html*** - Example.
- ***Full_Random_Presets_Dashboard.html*** - Example.
- ***FINAL_Dataset_Dashboard.html*** - Example.

**Other Files:**
- ***requirements.txt*** - Example.
- ***sylenth1_params.json*** - Example.
- ***sylenth_defaults.json*** - Example.
- ***user_presets.json*** - Example.
