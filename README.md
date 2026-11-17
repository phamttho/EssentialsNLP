# Robustness of Multilingual Intent Classification to Typos

This project studies how controlled character-level spelling corruptions affect multilingual intent classification on the MASSIVE 1.1 dataset. The experiment covers English (`en-US`), Vietnamese (`vi-VN`), German (`de-DE`), and Simplified Chinese (`zh-CN`).

## Project layout

```text
EssentialsNLP/
├── README.md
├── 01_data_processing/
│   ├── README.md
│   ├── EssentialsNLP.ipynb
│   └── exported_data/
├── 02_training/
└── 03_analysis/
```

`02_training/` and `03_analysis/` are planned project sections; their implementation and instructions will be added there.

## Part 1: Data processing

The processing notebook downloads the MASSIVE splits, prepares clean train/validation/test data, creates typo-augmented training examples and corrupted test sets, and exports metadata. For the detailed processing steps, CSV schemas, and exported folder structure, see [`01_data_processing/README.md`](01_data_processing/README.md).

### Run the data-processing notebook

The easiest way is to run it in PyCharm:

1. Open [`01_data_processing/EssentialsNLP.ipynb`](01_data_processing/EssentialsNLP.ipynb).
2. Select `01_data_processing/.venv/bin/python` as both the project interpreter and notebook kernel. Use Python 3.11 (the current environment is Python 3.11.9).
3. Set the notebook working directory to `01_data_processing`.
4. Choose **Run All**. The first code cell installs the notebook packages, so an internet connection is required.

If the environment does not exist, create a Python 3.11 virtual environment in `01_data_processing` through PyCharm's **Add Interpreter** option, then select it as the notebook kernel. The notebook must have a working interpreter before any cell can run.

## Part 2: Model training — placeholder

Add the training implementation and run instructions here. Planned scope: train multilingual intent classifiers on clean and typo-augmented data, with per-language configurations and reproducible seeds.

## Part 3: Evaluation and analysis — placeholder

Add evaluation and analysis implementation and run instructions here. Planned scope: compare clean and corrupted test performance, summarize robustness across languages and corruption conditions, and export metrics, tables, and figures.
