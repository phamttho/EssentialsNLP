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

Running the notebook creates only the original clean download CSVs in `typo_data/`:

- `{language}_train.csv`
- `{language}_validation.csv`
- `{language}_test.csv`

All quality summaries, overlap counts, typo-augmented data, corrupted test sets, and inspection pairs are displayed or held in memory only; they are not exported as additional files.
