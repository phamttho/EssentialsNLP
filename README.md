# Robustness of Multilingual Intent Classification to Typos

University of Zurich - Essentials in Text and Speech Processing, 2026

Authors: Ke Sun, Thi Tho Pham, Tristan Hein

This project studies the robustness of multilingual intent classification to controlled synthetic character corruptions using MASSIVE 1.1. We evaluate English (`en-US`), Vietnamese (`vi-VN`), German (`de-DE`), and Simplified Chinese (`zh-CN`) with word-level TF-IDF + Logistic Regression, character-level TF-IDF + Logistic Regression, and XLM-R-base.

The corruptions are controlled synthetic perturbations used as a proxy for typos. They are not validated models of realistic language-specific typing errors.

## Repository structure

```text
EssentialsNLP/
├── 01_data_processing/   # MASSIVE loading, cleaning, corruption generation, exported data
├── 02_training/          # model training, HPO, retained metrics/evidence, reproducibility docs
├── 03_analysis/          # evaluation, verification, figures, tables, qualitative analysis
├── report/               # final LaTeX report source and reproducible report figures
└── README.md
```

Each project part has its own README and dependencies.

## 01 - Data processing

Entry point:

```text
01_data_processing/EssentialsNLP.ipynb
```

Install the data-processing dependencies with:

```bash
pip install -r 01_data_processing/requirements.txt
```

Run the notebook from the `01_data_processing` directory. It produces the clean, augmented, and corrupted test exports under:

```text
01_data_processing/exported_data/
```

See `01_data_processing/README.md` for details.

## 02 - Model training

The training archive contains the original fixed-setting experiment and the supplementary validation-based hyperparameter search.

Three model families are evaluated separately for each language:

- word TF-IDF + Logistic Regression
- character TF-IDF + Logistic Regression
- XLM-R-base

Both clean-only and mixed clean/corrupted training are evaluated with seeds 2028, 2029, and 2030.

The lightweight archive verifier requires no GPU:

```bash
cd 02_training
python -B tools/verify_archive.py
```

Full retraining is not a one-command portable workflow. The archived experiment used a specific Linux/AutoDL layout, frozen inputs, pretrained files, and GPU environment. See `02_training/README.md` and `02_training/docs/REPRODUCIBILITY.md`.

The historical reproduction guide intentionally restores the frozen data checkout under `01_data/`; that path belongs to the recorded training environment and is separate from the final repository folder `01_data_processing/`.

## 03 - Evaluation and analysis

Install the analysis dependencies with:

```bash
pip install -r 03_analysis/requirements.txt
```

The analysis consumes the completed original prediction handoff and the frozen exported data:

```bash
python 03_analysis/run_analysis.py \
  --handoff /PATH/TO/handoff_03_20261016 \
  --data-root /PATH/TO/EssentialsNLP/01_data_processing/exported_data \
  --output 03_analysis/analysis_outputs
```

Add `--verify-metrics` to independently recompute all stored accuracy and macro-F1 values from the full prediction handoff.

The large per-example prediction handoff is not committed to Git. The generated report-ready tables, figures, integrity checks, and summary are included in `03_analysis/analysis_outputs/`.

## Report

The final report source is under `report/`.

The two report figures can be reproduced from the included three-seed summary with:

```bash
python report/make_figures.py
```

The LaTeX source is `report/main.tex`.

## Main experimental design

- Dataset: MASSIVE 1.1
- Languages: English, Vietnamese, German, Simplified Chinese
- Intent classes: 60
- Corruptions: deletion, substitution, duplication, adjacent-character swap
- Nominal noise levels: 5%, 10%, 20%
- Training conditions: clean and mixed
- Seeds: 2028, 2029, 2030
- Metrics: accuracy, macro-F1, and clean-to-corrupted macro-F1 drop

The explicit classical baseline is word-level TF-IDF + Logistic Regression trained on clean data.

## Reproducibility and limitations

The project uses fixed corruption seeds, frozen corrupted test sets, three training seeds, saved configurations, retained metric summaries, and independent metric verification.

The synthetic corruption scheme should not be interpreted as realistic language-specific typing behavior. It does not model keyboard-neighbor errors, German compound-specific spelling errors, Vietnamese input-method/diacritic errors, or Chinese IME/homophone confusions. The same frozen corruption realization is reused across model seeds, and mixed training contains additional training examples relative to clean-only training.

See the READMEs inside each project folder and the final report for the complete experimental and reproducibility details.
