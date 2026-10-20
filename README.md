# Robustness of Multilingual Intent Classification to Typos

This project investigates the **robustness of multilingual intent-classification models to spelling noise** using the MASSIVE 1.1 dataset.

We study how controlled character-level corruptions affect intent classification across different languages and model types, and whether training with typo-augmented data can improve robustness.

The character corruptions in this project are **controlled synthetic perturbations used as a proxy for typos**. They allow reproducible comparisons across models and noise levels, but they do not fully reproduce all language-specific human typing errors such as keyboard-neighbor mistakes, Vietnamese input-method errors, or Chinese IME/homophone errors. This is therefore considered a limitation and a possible direction for future work.

## Authors

- Ke Sun
- Thi Tho Pham
- Tristan Hein

## Dataset

We use [MASSIVE 1.1](https://huggingface.co/datasets/AmazonScience/massive), a multilingual natural-language-understanding dataset containing more than one million utterances and 60 intent classes.

For this project, we focus on four languages:

- English: `en-US`
- Vietnamese: `vi-VN`
- German: `de-DE`
- Simplified Chinese: `zh-CN`

The main fields used for intent classification are:

- `utterance`: the user's text request
- `intent`: the categorical intent label, such as `weather_query`, `alarm_set`, or `email_sendemail`

## Project structure

The project is divided into three main parts:

```text
EssentialsNLP/
│
├── EssentialsNLP.ipynb
├── exported_data/
│
├── 02_training/
│   └── model-training code and configuration
│
├── 03_analysis/
│   ├── run_analysis.py
│   ├── requirements.txt
│   ├── README.md
│   └── analysis_outputs/
│       ├── figures/
│       ├── tables/
│       ├── RESULTS_SUMMARY.md
│       └── integrity_checks.json
│
├── requirements.txt
└── README.md
```

Large generated training artifacts, checkpoints, and raw prediction files are not required to be stored in the repository.

## Part 1: Data processing

The data-processing pipeline:

1. Loads the selected MASSIVE languages.
2. Prepares the train, validation, and test data.
3. Checks the dataset structure and intent labels.
4. Generates controlled character-level corruptions.
5. Produces typo-augmented training data.
6. Produces corrupted test sets.
7. Saves metadata needed for reproducibility.

The corruption operations include:

- character deletion
- character substitution
- character duplication
- adjacent-character swapping

Different corruption conditions and noise levels are generated with fixed random seeds so that the experiments can be reproduced.

Augmentation is applied only to the **training data**. The corrupted test sets are kept separate and are used only for evaluation.

### Exported data

Running the data-processing notebook exports the data under:

```text
exported_data/
├── clean/
│   ├── en-US/{train,validation,test}.csv
│   ├── vi-VN/{train,validation,test}.csv
│   ├── de-DE/{train,validation,test}.csv
│   └── zh-CN/{train,validation,test}.csv
│
├── augmented/
│   ├── en-US_train_augmented.csv
│   ├── vi-VN_train_augmented.csv
│   ├── de-DE_train_augmented.csv
│   └── zh-CN_train_augmented.csv
│
├── corrupted_test/
│   └── {language}/{typo_type}_{noise_level}.csv
│
└── metadata/
    ├── intent_mapping.json
    └── data_manifest.json
```

The manifest records information such as dataset configuration, row counts, generated paths, corruption types, noise levels, and random seeds.

## Part 2: Model training

Three model types are compared:

1. **Word-level TF-IDF + Logistic Regression**
2. **Character-level TF-IDF + Logistic Regression**
3. **XLM-R-base**

The word-level TF-IDF + Logistic Regression model trained on clean data is used as the main classical baseline.

Models are trained separately for the four selected languages so that robustness differences between languages can be compared directly.

For each model type, we compare:

- training on clean data
- training on a mixture of clean and typo-augmented data

The development set is used for model selection and tuning. Test data is used only for final evaluation.

Multiple random seeds are used for training. Final results are reported using the **mean and standard deviation across seeds**, rather than selecting the best-performing seed.

See the documentation inside `02_training/` for the exact training configuration and implementation.

## Part 3: Evaluation and analysis

The analysis compares performance on clean and corrupted test data.

The main evaluation metrics are:

- accuracy
- macro-F1

Our main robustness measure is the **drop in macro-F1 from clean to corrupted input**.

The analysis investigates:

- differences between the three model types
- differences between languages
- differences between corruption types
- the effect of increasing noise levels
- the effect of typo-augmented training
- whether robustness gains come at the cost of clean performance
- representative prediction errors
- commonly confused intents
- examples where augmentation recovers predictions that fail under clean-only training

All reported results are aggregated across the three training seeds.

The analysis also independently recomputes evaluation metrics from the saved prediction files to verify the training results.

### Analysis outputs

The generated analysis can be found under:

```text
03_analysis/analysis_outputs/
├── RESULTS_SUMMARY.md
├── integrity_checks.json
│
├── figures/
│   ├── 01_clean_vs_noisy_macro_f1.png
│   ├── 02_augmentation_noisy_gain.png
│   ├── 03_xlmr_mixed_noise_curve.png
│   ├── 04_xlmr_mixed_drop_by_typo_type.png
│   ├── 05_xlmr_mixed_robustness_by_language.png
│   └── 06_clean_trained_model_comparison.png
│
└── tables/
    ├── 01_by_language.csv
    ├── 02_overall_equal_language_weight.csv
    ├── 03_clean_model_baseline_comparison.csv
    ├── 04_augmentation_effect.csv
    ├── 05_by_typo_type.csv
    ├── 06_by_noise_level.csv
    ├── 07_worst_condition_per_system.csv
    ├── 08_xlmr_mixed_error_examples.csv
    ├── 09_xlmr_mixed_top_confusions.csv
    ├── 10_xlmr_augmentation_rescue_examples.csv
    ├── 11_metric_verification.csv
    └── 12_corruption_audit.csv
```

## Main observations

The experiments show several clear patterns.

Character-level TF-IDF is substantially more robust to character corruption than the word-level TF-IDF baseline, despite having similar performance on clean input.

XLM-R achieves much stronger performance on clean data, but the version trained only on clean input experiences a considerable performance drop under corruption.

Training XLM-R with typo-augmented data substantially improves its performance on corrupted inputs while largely preserving its clean-input performance.

The robustness effects also vary across languages, corruption types, and noise levels.

These findings should be interpreted specifically in the context of the **controlled synthetic corruption scheme used in this project**. They do not imply that one language is inherently more robust to real-world typing errors than another.

## Setup

Install the dependencies for the data-processing part with:

```bash
python3 -m pip install -r requirements.txt
```

Then open:

```text
EssentialsNLP.ipynb
```

and run the cells from top to bottom.

The notebook creates the files in `exported_data/`.

Model training is documented separately under:

```text
02_training/
```

The analysis has its own lightweight requirements:

```bash
python3 -m pip install -r 03_analysis/requirements.txt
```

See:

```text
03_analysis/README.md
```

for the exact analysis command and expected input structure.

## Reproducibility

The project uses:

- fixed random seeds for corruption generation
- consistent train, validation, and test data
- identical corrupted test examples across models
- multiple model-training seeds
- saved experiment settings
- mean and standard deviation across seeds
- independent metric verification from prediction files

These choices are intended to make the experiments and analysis reproducible.

## Limitations

The corruption generator provides a controlled way of evaluating robustness, but it does not fully model real-world language-specific typing behavior.

For example, realistic noise may involve:

- keyboard-adjacent key errors
- German-specific spelling and compound-word errors
- Vietnamese diacritic and input-method errors
- Chinese IME, homophone, or character-selection errors

A possible extension of this project would therefore be to construct language-specific test sets containing more realistic human typing mistakes and compare whether the conclusions from synthetic corruption also hold in those conditions.

## Course

University of Zurich  
**Essentials in Text and Speech Processing, 2026**
