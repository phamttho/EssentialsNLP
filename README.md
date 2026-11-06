# Multilingual intent classification under synthetic typos

This branch contains Ke Sun's **model-training work (Part 02)** for the Essentials in Text and Speech Processing project. It includes the original experiment and the supplementary hyperparameter search, together with their configurations, training histories and result summaries.

The project asks how character-level spelling noise affects intent classification across languages, and whether training with corrupted examples improves robustness. The included results are completed experiments; this publication does not claim that tuning improved every model or condition.

## Start here

- **Understand the experiment:** read the design below and [the original experiment guide](docs/original_run_README.md).
- **Inspect the original results:** see `outputs/summaries/`, `outputs/test_metrics/` and `outputs/validation_metrics/`.
- **Inspect tuning decisions:** see `hyperparameter_tuning/results/search_validation_scores.csv` and `hyperparameter_tuning/results/selection.json`.
- **Compare original and tuned results:** see `hyperparameter_tuning/results/summaries/baseline_vs_tuned.csv`.
- **Plot training curves:** use the retained training-history JSON files described below.
- **Check this publication:** run `python -B tools/verify_archive.py` from the repository root. This uses the Python standard library and needs no GPU.
- **Run or verify the archived code:** first read [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). This is an execution archive, not a portable one-command training package.

## Experimental design

| Item | Setting |
|---|---|
| Dataset | Frozen MASSIVE 1.1 exports from the data-processing part |
| Languages | English (`en-US`), Vietnamese (`vi-VN`), German (`de-DE`), Simplified Chinese (`zh-CN`) |
| Task | Intent classification with 60 shared intent labels |
| Training | Separate models for each language; no joint multilingual training |
| Models | Word TF-IDF + Logistic Regression; character TF-IDF + Logistic Regression; fine-tuned XLM-R-base |
| Conditions | `clean`, or `mixed`: one clean and one corrupted copy per training example |
| Seeds | `2028`, `2029`, `2030` |
| Evaluation | Clean test data and 12 corrupted test conditions: four typo operations at three noise levels |
| Metrics | Accuracy, macro-F1 and the clean-to-noisy performance drop |

The original experiment contains **72 runs**: four languages × three models × two conditions × three seeds. Word TF-IDF with Logistic Regression trained on clean data is the explicit baseline. Original Logistic Regression used `C=1`; original XLM-R used a learning rate of `2e-5`.

XLM-R was allowed up to 30 epochs, with early stopping after 10 validation evaluations without an improvement in clean-validation macro-F1. The best validation model was used for prediction. Logistic Regression uses its own iterative solver, with `max_iter=1000`; its iterations are not XLM-R epochs.

## Supplementary hyperparameter search

- Logistic Regression: `C ∈ {0.1, 1, 10}`, independently for the two TF-IDF models.
- XLM-R: learning rate `∈ {1e-5, 2e-5, 3e-5}`.
- For each language/model pair, select one value shared by `clean` and `mixed` training.
- Selection maximizes the mean **clean-validation macro-F1 across all six runs**: two training conditions × three seeds. Exact ties prefer the original parameter.
- Candidate test scores are not used for selection. Original test results had already been inspected, so this is reported as a supplementary experiment.

The search contains **216 candidate records**: 72 reused original validation results and 144 newly trained candidates. After selection, there are **72 final evaluations**: 60 newly trained runs and 12 reused original runs. Each final run has 13 test conditions, producing **936 metric rows** and **312 groups summarized across three seeds**.

## What is in this branch?

```text
configs/                         Original experiment settings
src/, scripts/, tests/           Original training, evaluation and tests
requirements*.txt                Recorded dependencies
outputs/                        Original metrics and run evidence
evidence/                       Archived histories and selected execution evidence
provenance/                     Data/model metadata and source-file inventory
hyperparameter_tuning/          Supplementary code, settings and retained results
docs/                           Original guide and reproduction requirements
```

The data-processing notebook and exported datasets are not duplicated at this branch's root. The frozen data source is repository commit `6f6bd31ad98edc64bee375a203f516cb70efbcda`; see the reproduction guide for restoring it as a separate `01_data` checkout.

Large data files, installed environments, caches, trained weights and full prediction files are excluded from Git. The complete predictions remain in the private project storage. Contact **Ke Sun** for the corresponding result files or private handoff package; this repository does not provide a public download link.

## Reading results and plotting curves

- Metric values are stored on a **0–1 scale**, not as percentages.
- Three-seed summaries use the arithmetic mean and sample standard deviation (`ddof=1`). Keep all three seeds; do not choose only the best seed.
- Performance drops are computed as clean minus noisy performance within each seed, then summarized across seeds.
- `baseline_vs_tuned.csv` compares the selected setting against the original fixed setting for the same language, model, training condition, seed and test condition. It is separate from the project's word-TF-IDF clean-training baseline.
- Original XLM-R histories are under `evidence/original_training_histories/`. New search and refit histories are retained under `hyperparameter_tuning/search/` and `hyperparameter_tuning/results/final/`.
- XLM-R histories record training loss at logging steps and validation loss, accuracy and macro-F1 at epoch evaluations. **Per-epoch training accuracy was not recorded.** Missing metrics cannot be reconstructed from loss alone.
- Histories can be plotted locally or imported into a visualization service later. No W&B account is required to inspect the files. TF-IDF models do not have equivalent epoch-by-epoch training curves.

## Interpretation and limitations

The experiments use the teammate's frozen, cleaned exports. They are **not unchanged official MASSIVE splits**: preprocessing removes duplicate training examples and certain normalized overlaps across splits, while preserving the test split. The recorded `source_row` identifies a row in an exported clean split, not an original MASSIVE identifier.

The typo generator produces controlled synthetic character deletion, substitution, duplication and swaps at levels `0.05`, `0.10` and `0.20`. These are not validated models of realistic keyboard or IME errors. The character pool was constructed using held-out text as well as training text, and overlapping swaps are possible. Short texts, unchanged outputs and language differences should be considered when interpreting robustness results.

The same frozen augmented data and test corruptions are used across model seeds. The supplementary search changes model hyperparameters; it does not repair these data-generation limitations. Refit validation scores can differ slightly from search scores, and those differences are recorded.

## Weights and reproducibility

New tuning weights were deleted only after the required predictions had been exported and verified. Training histories, metrics and provenance were retained. New predictions for previously unseen input would require retraining a selected tuned model.

The scripts preserve their original Linux/AutoDL paths and fingerprint checks. Restoring datasets, pretrained files, dependencies and any omitted prediction artifacts is necessary before executing them. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact requirements.
