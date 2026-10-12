# MASSIVE 1.1 Typos

This project studies the **robustness of multilingual intent classification to typos**. It investigates how realistic character-level spelling errors affect intent-classification systems across languages and model types, and whether typo augmentation improves robustness.

The project uses MASSIVE 1.1 and is organized into three parts:

1. Data processing
2. Model training
3. Evaluation and analysis

## Languages
For this project we chose :

- English: `en-US`
- Vietnamese: `vi-VN`
- German: `de-DE`
- Simplified Chinese: `zh-CN`

Only the `utterance` and `intent` fields are retained in the original CSV files. `utterance` is the user’s text request. `intent` is a categorical class ID representing the user’s goal, such as `weather_query`, `alarm_set`, or `email_sendemail`; it is not a numerical score.

## Setup

From this project directory:

```bash
python3 -m pip install -r requirements.txt
```

Open `EssentialsNLP.ipynb` and run the cells from top to bottom.

## Output files

Running the notebook exports all data under `exported_data/`:

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
│   └── {language}/{typo_type}_{noise_level}.csv
└── metadata/
    ├── intent_mapping.json
    └── data_manifest.json
```

The manifest records the dataset configuration, row counts, generated paths, typo types, noise levels, and reproducibility seed.
