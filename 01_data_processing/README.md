# Data processing: MASSIVE typo robustness dataset

This folder prepares data for multilingual intent classification experiments. The pipeline downloads the official [MASSIVE 1.1](https://huggingface.co/datasets/AmazonScience/massive) splits for English (`en-US`), Vietnamese (`vi-VN`), German (`de-DE`), and Simplified Chinese (`zh-CN`), then exports clean, typo-augmented, and synthetically corrupted CSV files.

The notebook is the data-processing entry point: [`EssentialsNLP.ipynb`](EssentialsNLP.ipynb). It retains only the fields needed for intent classification in the clean data: `utterance` and `intent`. The `intent` field is the dataset's numeric class ID; names for those IDs are in the exported intent mapping.

## Run the pipeline

### Easiest: run it in PyCharm

1. Open `EssentialsNLP.ipynb`.
2. Set the project interpreter and notebook kernel to `01_data_processing/.venv/bin/python` (Python 3.11.9).
3. Set the notebook working directory to `01_data_processing` so exports are written to this folder's `exported_data/` directory.
4. Choose **Run All**. The first code cell installs the notebook's Python packages into the selected kernel; an internet connection is needed.

If PyCharm reports `Cannot run program .../.venv/bin/python` or `No such file or directory`, it is pointing to a missing or incorrect interpreter. Browse to `.venv/bin/python` inside the actual `01_data_processing` folder and select it for both the project interpreter and notebook kernel. Notebook cells cannot fix a missing interpreter because the kernel must start before any cell can execute.

### If the environment is missing

Create a Python 3.11 virtual environment in `01_data_processing` using PyCharm's **Add Interpreter** option, then select its `.venv/bin/python` for both the project interpreter and notebook kernel. If you create the environment from a terminal instead, run `python3.11 -m venv .venv` from this folder. After PyCharm can start the kernel, use **Run All**; the notebook setup cell installs its packages.

## Export structure

After the notebook finishes, `exported_data/` has this structure:

```text
exported_data/
├── clean/
│   ├── en-US/{train,validation,test}.csv
│   ├── vi-VN/{train,validation,test}.csv
│   ├── de-DE/{train,validation,test}.csv
│   └── zh-CN/{train,validation,test}.csv
├── augmented/
│   ├── en-US_train_augmented.csv
│   ├── vi-VN_train_augmented.csv
│   ├── de-DE_train_augmented.csv
│   └── zh-CN_train_augmented.csv
├── corrupted_test/
│   └── {en-US,vi-VN,de-DE,zh-CN}/
│       └── {deletion,substitution,duplication,swap}_{0.05,0.10,0.20}.csv
└── metadata/
    ├── intent_mapping.json
    └── data_manifest.json
```
