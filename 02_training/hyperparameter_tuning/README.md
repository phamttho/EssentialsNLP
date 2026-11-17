# Supplementary hyperparameter tuning
This independent experiment does not modify the original 72 runs.
Everything written by this experiment stays in this directory. The existing
environment, input datasets and pretrained snapshot are used read-only.

## Fixed study design
- Four separate languages: en-US, vi-VN, de-DE, zh-CN.
- Word and character TF-IDF + Logistic Regression: C in [0.1, 1, 10].
- XLM-R-base: learning_rate in [1e-5, 2e-5, 3e-5].
- Conditions: clean and mixed (same frozen 50/50 exports as the original).
- Seeds: 2028, 2029, 2030. All other training settings remain unchanged.
- XLM-R max30 epochs, patience10, clean validation macro-F1, batch16,
  accumulation2, max_length64, bf16, AdamW, linear schedule, no warmup.
- Per language/model select one value shared by clean and mixed. The criterion
  is arithmetic mean of SIX clean validation macro-F1 scores: 2 conditions x
  3 seeds. Exact ties prefer the original parameter, then declared grid order.
- Candidate grid and criterion are fixed before launching. Original test results
  have already been seen, so report this as supplementary tuning, not a
  previously untouched confirmatory study.
- Selection uses only validation scores, never candidate test scores.
- Fixed synthetic noise, split cleaning, held-out character-pool and overlapping
  swap limitations from the original study remain. This experiment does not fix
  the noise generator or change the data.

## Workload and stages
1. Read-only preflight verifies all input hashes, pretrained hashes, original
   run fingerprint and completed validation records.
2. Reuse 72 original validation records; fit 144 new candidates (96 LR + 48 XLM-R).
3. After all 216 scores are complete, lock 12 language/model parameter choices.
4. The final stage completed **60 new training runs and 12 reuses of original
   runs**, for **72 final evaluations**. The 60 new runs comprised 48 runs
   with TF-IDF and Logistic Regression at `C=10` and 12 XLM-R runs in
   English and Simplified Chinese at `3e-5`. German and Vietnamese XLM-R
   retained `2e-5`, so their 12 original runs supplied the verified test
   predictions.
5. All 72 final evaluations were verified on the clean test set and 12
   corruption conditions. The final tables contain 936 metric rows and 312
   three-seed groups. Final predictions were exported as compressed CSV files.
A refit can differ numerically from the search run. Its actual validation score,
best epoch and difference from the search score are recorded; there is no
test-based reselection or claim of bit-identical reproduction.

## Storage policy
Three independent GPU workers (physical0,1,2), two CPU workers, four CPU threads
per worker. GPU workers dynamically take jobs; four languages do not require
four GPUs. Every worker exposes exactly one GPU, no distributed training.
Only one BEST weights-only file per active fit; atomic updates may temporarily
hold both old and new files. Optimizer/scheduler/RNG recovery checkpoints are
disabled. The caller verifies reloaded validation predictions before deleting
search weights. For selected refits, it verifies all test predictions first.
Permanent: settings, source, logs, histories, validation predictions, selected
parameters, metrics, final predictions and checksums. The completed tuning
experiment retained these artifacts after removing its new weights. The tuning
runner preserved the original models during the experiment. A separate storage
cleanup later removed the 24 original XLM-R weight files after the prediction
backup passed verification. The saved predictions and training records remain
available for analysis.
An interrupted fit restarts from scratch, but completed candidate/final records
are verified and skipped. Failed task work is retained until explicit restart.
Only work/<specific_job>/ within this new directory can be cleaned by the runner.

Disk floor25GiB plus conservative outstanding reservations (3.5GiB per GPU job,
0.5GiB per CPU job). Full-concurrency startup threshold36.5GiB. Insufficient
space produces a safety stop; it never cleans other projects. This guard is
conservative and cannot reserve disk against unrelated external processes.

## Install and verify the environment

Follow the [reproduction guide](../docs/REPRODUCIBILITY.md#install-and-verify-the-locked-environment)
to install the shared training environment from `../requirements.lock.txt` and
verify the archived results. The [original tuning README](../docs/original_hpo_README.md)
preserves the instructions recorded during the experiment.

## Start in background
    cd /root/autodl-tmp/essentials_in_text_and_speech/02_training/hyperparameter_tuning
    bash scripts/start.sh

The runner detaches from SSH and inherits a process lock through child tasks.
Do not start a second copy or edit source/config while any task is active.
Check live process and log progress after starting before closing your laptop.
If a task fails, read its log; after resolving the cause, run the same start
command. Completed work is reused if integrity and fingerprint checks pass.
If changing the scientific code/config, use a new experiment directory/version.

## One-shot status
    bash scripts/status.sh

## Main outputs
- plan.json: frozen study definition.
- state/preflight.json: verified source/environment information.
- state/experiment.json: locked code/config/evidence fingerprint.
- records/search/: 216 candidate summaries after completion.
- search/<candidate>/: validation metrics, predictions and training histories
  (original imports have metrics and provenance, not new validation inference).
- results/search_validation_scores.csv: all 216 validation scores.
- results/selection.json: 12 shared parameter selections locked before testing.
- results/final/<candidate>/: predictions.csv.gz, metrics.json, verification.json
  and final refit history/validation evidence or legacy source information.
- records/final/: 72 tested-run summaries and hashes.
- results/summaries/: all_test_metrics.csv and three_seed_summary.csv.
- logs/: runner and per-task logs, with epoch evaluation progress.
CSV metrics use0to1, not percentages. Sample SD uses ddof1 across three seeds.
Preserve all three seeds; do not pick the best seed.
If a chosen model must later predict NEW inputs, retraining is necessary because
new final weights are intentionally removed after export verification.

## Safety
No downloading, no dependency installation and no changes to shared Python.
No old result, cache, dataset or other project is moved or deleted.
Starting requires all selected GPUs to be free; each GPU is checked again
before dispatch. A low utilization percentage alone does not mean free.
Deployment self-checks use synthetic tiny CPU training and existing-file
verification, not production training. Formal tasks start only with start.sh.
