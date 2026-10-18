# 03 Analysis: MASSIVE typo robustness

This folder is Tristan's Part 03. It is designed to use Sun's **finished outputs only**. It does **not** retrain TF-IDF or XLM-R and does not require a GPU.

## What Part 03 is responsible for

1. Verify the handoff is complete and internally consistent.
2. Use the mean and sample standard deviation across seeds 2028, 2029, and 2030. Never choose the best seed.
3. Compare the explicit baseline, **word TF-IDF + Logistic Regression trained on clean data**, with character TF-IDF and XLM-R.
4. Compare clean training with mixed/typo-augmented training.
5. Measure clean performance, mean corrupted-test performance, and the clean-to-noisy macro-F1 drop.
6. Compare languages, corruption types, and noise levels.
7. Do row-level error analysis from the saved predictions, including top confusion pairs and examples where corruption breaks a previously correct prediction.
8. Identify examples where mixed XLM-R fixes errors made by clean-trained XLM-R.
9. Produce report-ready tables, figures, and a short results/limitations summary.

## Where to put this in the team GitHub

Add this directory as `03_analysis/` in the same repository as Thi Tho's data-processing work.

Do **not** commit Sun's full `outputs/predictions/` folder to normal Git. It is roughly 500 MB and is an analysis input, not source code. Keep Sun's handoff extracted locally, or store it using whatever large-file sharing method the group uses. Commit the analysis code plus the small generated tables/figures.

A clean team layout can look like:

```text
EssentialsNLP/
├── EssentialsNLP.ipynb           # Part 01, data processing
├── exported_data/                # Part 01 exports
├── 03_analysis/                  # this folder
│   ├── run_analysis.py
│   ├── requirements.txt
│   └── analysis_outputs/         # generated small tables/figures
└── README.md
```

Sun's handoff can live outside the repo, for example:

```text
~/project_inputs/handoff_03_20261016/
```

## Run it

Create/activate any normal Python environment, then:

```bash
pip install -r 03_analysis/requirements.txt
python 03_analysis/run_analysis.py \
  --handoff /PATH/TO/handoff_03_20261016 \
  --data-root /PATH/TO/EssentialsNLP/exported_data \
  --output 03_analysis/analysis_outputs
```

This uses CPU only.

To independently recompute all 936 saved metric records from Sun's prediction files, add:

```bash
--verify-metrics
```

That check reads the full prediction handoff, so it is slower but still does not train anything.

## Main generated files

### Tables
- `01_by_language.csv`: clean/noisy macro-F1 and drop, mean ± std across seeds.
- `02_overall_equal_language_weight.csv`: compact overall comparison.
- `03_clean_model_baseline_comparison.csv`: explicit comparison to the clean word-TF-IDF baseline.
- `04_augmentation_effect.csv`: clean-score change, noisy-score gain, and drop reduction from mixed training.
- `05_by_typo_type.csv`: corruption-type comparison.
- `06_by_noise_level.csv`: nominal noise-level comparison.
- `07_worst_condition_per_system.csv`: lowest-F1 corruption for each language/model/training condition.
- `08_xlmr_mixed_error_examples.csv`: paired clean/noisy error examples using majority vote across seeds.
- `09_xlmr_mixed_top_confusions.csv`: top intent confusions in the worst XLM-R mixed condition per language.
- `10_xlmr_augmentation_rescue_examples.csv`: cases fixed by mixed training.
- `12_corruption_audit.csv`: actual edit-distance/unchanged/empty counts when `--data-root` is provided.
- `11_metric_verification.csv`: only created with `--verify-metrics`.

### Figures
The script produces six separate report-ready figures in `analysis_outputs/figures/`.

### Notes
Read `analysis_outputs/RESULTS_SUMMARY.md`. It summarizes the key descriptive findings and, importantly, the limitations that should not be hidden in the final report.

## Important scientific boundary

The current typo generator should be described as **controlled synthetic character corruption**, not as a validated simulation of natural human misspellings. The existing generator does not implement keyboard-neighbor errors, Vietnamese input-method/diacritic errors, Chinese IME/homophone errors, or German-specific spelling/compound errors. This matters because the course feedback explicitly asked the team to think carefully about realistic typos by language.

The existing results are still useful for the project's core question about robustness to controlled character corruption. If the team wants to address the feedback more strongly, the most efficient extension is to create a small language-specific realistic **test-only** benchmark and ask Sun to run inference with the already trained models. That would not require Tristan to retrain anything.
