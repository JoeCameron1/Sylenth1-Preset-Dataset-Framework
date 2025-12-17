# Sylenth1-Preset-Dataset-Framework

## Dataset & Code Repository for the paper "A Dataset and Analysis Framework for Large-Scale Synthesizer Preset Generation"

This repository contains a dataset of presets from the Sylenth1 synthesizer along with the AudioCommons timbral model values for each preset. It also contains all the code used to create the dataset and for further analysis.

Below is an explanation of each file in this repository.

**Dataset Files:**
- ***[Sylenth1_Full_Factory_Presets_AC_A4.json](Sylenth1_Full_Factory_Presets_AC_A4.json)*** - This contains the 292 factory presets of Sylenth1 along with their corresponding AudioCommons timbral model values.
- ***[Full_Random_Presets.json](Full_Random_Presets.json)*** - This contains the 10,000 randomly generated presets from Sylenth1 along with their corresponding AudioCommons timbral model values.
- ***[FINAL_timbral_dataset_audiocommons.json](FINAL_timbral_dataset_audiocommons.json)*** - This contains the total combined dataset of the 292 factory presets from [Sylenth1_Full_Factory_Presets_AC_A4.json](Sylenth1_Full_Factory_Presets_AC_A4.json) and the 10,000 randomly generated presets from [Full_Random_Presets.json](Full_Random_Presets.json) along with their corresponding AudioCommons timbral model values.
- ***[random_preset_audio_snippets/](random_preset_audio_snippets/)*** - This folder contains the audio snippets for the 10,000 randomly generated presets used to calculate their corresponding AudioCommons timbral model values.

**Code Files:**
- ***[sylenth1_preset_custom_app.py](sylenth1_preset_custom_app.py)*** - This is the custom app used to interact with and control Sylenth1. Presets were created and saved to the dataset via this app. *Usage*: `python sylenth1_preset_custom_app.py`
- ***[build_sylenth1_timbre_dashboard.py](build_sylenth1_timbre_dashboard.py)*** - This builds a Plotly dashboard for analysing the contents and distributions of the dataset's .JSON files. The distribution and correlation figures from Sections 5.1 & 5.2 in the paper are produced by this script. *Usage*: `python build_sylenth1_timbre_dashboard.py --in [TIMBRAL_DATASET].json --out [DASHBOARD].html` (you may have to `pip install pandas numpy plotly scikit-learn` on the virtual environment)
- ***[pca_audiocommons_analysis.py](pca_audiocommons_analysis.py)*** - This performs all the PCA analysis on the dataset's .JSON files. The PCA figures from Section 5.3 in the paper are produced by this script. *Usage*: `python pca_audiocommons_analysis.py --in [TIMBRAL_DATASET].json --outdir [PCA_ANALYSIS_DIRECTORY]` (you may have to `pip install pandas numpy scikit-learn matplotlib` on the virtual environment)
- ***[gen_ac_hist.py](gen_ac_hist.py)*** - This script produces a set of histograms of the AudioCommons timbral model values across the presets' values. The histogram figure from Section 5.1 in the paper is produced by this script. *Usage*: `python gen_ac_hist.py` (the input dataset path is set in the script's code) (you may have to `pip install matplotlib` on the virtual environment)
- ***[gen_pca_rand_vs_fact.py](gen_pca_rand_vs_fact.py)*** - This script produces a PCA overlay comparison between two different datasets. The PCA overlay figures from Section 5.4 in the paper are produced by this script. *Usage*: `python gen_pca_rand_vs_fact.py` (takes the combined 10,292 dataset JSON file and this is set in the script's code) (you may have to `pip install pandas numpy scikit-learn matplotlib` on the virtual environment)
- ***[Sylenth1_Full_FactoryPresets_Timbre_Dashboard.html](Sylenth1_Full_FactoryPresets_Timbre_Dashboard.html)*** - The Plotly dashboard for the 292 factory presets of Sylenth1 produced by [build_sylenth1_timbre_dashboard.py](build_sylenth1_timbre_dashboard.py).
- ***[Full_Random_Presets_Dashboard.html](Full_Random_Presets_Dashboard.html)*** - The Plotly dashboard for the 10,000 randomly generated presets produced by [build_sylenth1_timbre_dashboard.py](build_sylenth1_timbre_dashboard.py).
- ***[FINAL_Dataset_Dashboard.html](FINAL_Dataset_Dashboard.html)*** - The Plotly dashboard for the total dataset (combination of 292 factory presets plus 10,000 randomly generated presets) produced by [build_sylenth1_timbre_dashboard.py](build_sylenth1_timbre_dashboard.py).

**Other Files:**
- ***[requirements.txt](requirements.txt)*** - This contains the pip dependencies for the custom Sylenth1 controller app ([sylenth1_preset_custom_app.py](sylenth1_preset_custom_app.py)). Run `pip install -r requirements.txt` in a virtual environment.
- ***[sylenth1_params.json](sylenth1_params.json)*** - This contains all the relevant parameters of Sylenth1 for this dataset, including their value ranges and descriptions.
- ***[sylenth_defaults.json](sylenth_defaults.json)*** - This contains the default parameter values for Sylenth1's initial setting (the INIT preset) on startup.
- ***[user_presets.json](user_presets.json)*** - This contains the saved user presets from the custom Sylenth1 controller app ([sylenth1_preset_custom_app.py](sylenth1_preset_custom_app.py)).
