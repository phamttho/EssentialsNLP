Final selected-parameter analysis package
all_test_metrics.csv: 936 records, one per selected run and test condition.
three_seed_summary.csv: 312 groups with mean and sample standard deviation (ddof=1).
final_training_summary.csv: 72 selected runs, provenance, validation scores and actual/best epochs.
baseline_vs_tuned.csv: 312 groups comparing each tuned result with its original fixed-parameter counterpart.
All accuracy/F1 values are fractions, not percentages. Multiply by 100 for percentage-point reporting.
All *_delta_* fields mean tuned minus original. A positive macro_f1_delta_mean is improvement.
For *_drop_delta_*, a negative value means the clean-to-noisy drop became smaller.
Delta standard deviations are computed across the three paired seed differences, not from unpaired means.
Within every language/model, clean and mixed share the chosen parameter.
The original clean-trained word TF-IDF model remains the project's explicit main baseline.
The baseline_vs_tuned table also compares every other model/condition with its own original fixed setting.
refit_validation_difference is the final refit's validation F1 minus its search validation F1.
For reused original runs, this difference is zero and no new refit was performed.
final_stage_wall_minutes includes import or training plus test export; it is not pure training time.
Temporary new model weights are not an analysis requirement; verified row-level predictions are retained.
