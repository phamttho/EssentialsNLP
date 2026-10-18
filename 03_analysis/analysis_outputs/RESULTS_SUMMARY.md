# Part 03 analysis notes

These notes are generated from Sun's frozen training handoff. They are descriptive, not significance-test claims.

## What was checked
- 72 completed run groups, 72 prediction files, and 936 run-condition metric rows.
- Every run has 13 test conditions: clean plus 12 corrupted conditions.
- Report the mean and sample standard deviation across seeds 2028, 2029, and 2030. Do not select a best seed.

## Main findings
- Explicit baseline, word TF-IDF + Logistic Regression trained on clean data: 69.09 clean macro-F1 and 50.33 mean corrupted-test macro-F1 (equal-weight average over the four languages).
- Character TF-IDF trained on clean data is much more robust than the word baseline: 60.32 mean corrupted-test macro-F1, with a 10.05-point clean-to-noisy drop versus 18.76 points for the word baseline.
- Clean-trained XLM-R has the strongest clean score (82.51) but a large average robustness drop (19.48 points). Its noisy average (63.03) is only slightly above clean-trained character TF-IDF (60.32).
- Mixed/augmented XLM-R is the strongest system on corrupted inputs: 74.57 mean corrupted-test macro-F1, with an 8.03-point drop.
- The largest XLM-R noisy gain from augmentation is in English: +16.72 macro-F1 points.
- Augmentation is not free: English XLM-R clean macro-F1 changes by -0.80 points while its noisy macro-F1 changes by +16.72 points.
- For mixed XLM-R, substitution is the most damaging corruption type on average across languages/levels, with a mean drop of 11.24 points.
- Increasing the nominal corruption level generally hurts performance, but Simplified Chinese is not strictly monotonic at 5% vs 10% in the current fixed synthetic corruptions.

## Important limitations to state in the report
- The current generator is controlled synthetic character corruption, not a validated model of natural human typos. It does not model keyboard adjacency, Vietnamese input-method/diacritic errors, Chinese IME/homophone errors, or German compound spelling.
- The substitution character pools were built from all retained splits, including held-out text. Labels are not used, but the corruption process is therefore not strictly train-text-only.
- Swap selection can use overlapping adjacent pairs. Do not describe swaps as guaranteed disjoint adjacent transpositions.
- Some rows marked corrupted are unchanged, and a few deletion outputs are empty.
- The same frozen noisy datasets are reused for all three training seeds. Standard deviations therefore describe training variability conditional on one corruption realization, not uncertainty over possible typo samples.
- Mixed training doubles the number of training examples. Its gain cannot be attributed purely to augmentation independent of extra optimization exposure.
- No hyperparameter search, compute-matched augmentation control, truncation audit, confidence intervals, or significance tests were run.

## Strongest report story
1. Start from the explicit word-level TF-IDF + Logistic Regression clean baseline.
2. Show that character features are substantially more robust to character corruption than word features.
3. Show that XLM-R is strongest on clean data but clean-trained XLM-R can be surprisingly brittle under synthetic corruption.
4. Show that typo-augmented/mixed training strongly improves XLM-R robustness, while noting the small clean-performance trade-off in English.
5. Finish with language/corruption differences and a careful limitations discussion, especially the realism issue raised by the course feedback.

## Corruption audit from exported data
- Across all language/type/level files, 175 generated rows are textually unchanged and 6 noisy rows are empty.
- See `tables/12_corruption_audit.csv` for actual edit-distance summaries by language, operation and nominal level.
