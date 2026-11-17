# Independent multilingual intent-classification training

Scope: 02 training only. Inputs are read from ../01_data/exported_data.
All environments, caches, temporary files, models, logs and results stay inside this project.
The upstream data is frozen as exported; its original split/provenance is recorded in the data manifest.

## Experiment
- Four separately trained languages: en-US, vi-VN, de-DE, zh-CN. No joint multilingual training.
- Word TF-IDF + Logistic Regression, character TF-IDF + Logistic Regression, XLM-R-base.
- Clean training and mixed training (one clean plus one augmented example per source row).
- Seeds 2028, 2029, 2030: 72 runs, including 48 CPU TF-IDF runs and 24 GPU XLM-R runs.
- GPU 0: English, GPU 1: Vietnamese, GPU 2: German, GPU 3: Chinese.
  Each GPU runs one model at a time. Two CPU queues run alongside them.
- XLM-R: at most 30 epochs, patience 10 evaluations, minimum improvement 0.
  Evaluate/save after each epoch. Select highest clean-validation macro-F1;
  a tie is not an improvement. Reload the best weights before all test evaluations.
- XLM-R batch 16, gradient accumulation 2 (effective batch 32), maximum length 64,
  BF16, AdamW, learning rate 2e-5, weight decay 0.01, linear decay over the
  configured 30-epoch horizon, no warmup.
- LR uses lbfgs, C=1, max_iter=1000 and must converge.
  TF-IDF fits only training data. Chinese word features use jieba.
  These deterministic baselines may produce identical results across seeds.
  LR iterations are not neural-network epochs.
- All runs use identical frozen clean and corrupted test sets (13 conditions).
  No test-based model selection and no selection of the best seed.

## Start (on the server)
```bash
cd /root/autodl-tmp/essentials_in_text_and_speech/02_training
bash scripts/start.sh
```

The runner detaches from SSH. It checks all four GPUs before starting and checks a
GPU again before each new task. Occupied GPUs cause refusal, even at low utilization.
Do not run other jobs on these GPUs while this experiment is active.
The start command prevents duplicate project runners.

One-shot progress:
```bash
/root/autodl-tmp/essentials_in_text_and_speech/02_training/environment/bin/python -B /root/autodl-tmp/essentials_in_text_and_speech/02_training/scripts/status.py
```

Confirm the runner/worker is alive and logs show progress before disconnecting.
This does not shut down or release the server.

## Resume and storage
Run the same start command after an interruption. Verified completed runs are skipped.
XLM-R resumes from the latest complete epoch checkpoint, including optimizer,
scheduler, random state and early-stopping patience counter. Work since that
checkpoint may repeat. TF-IDF restarts if no completed artifact exists.
Code/config/data fingerprints prevent mixing experiments after edits.

Keep two recent/best training checkpoints per active XLM-R run. Once the final
best model and prediction artifacts have been saved and verified, only that
run's temporary checkpoints are removed. Final model weights remain.
Do not manually remove checkpoints of unfinished tasks.

## Outputs
- models/<language>/<model>/<condition>/seed<seed>/: final saved models and training history.
- logs/: runner log and one log per experiment.
- outputs/manifests/: data, environment, run states, final handoff record.
- outputs/validation_metrics/: best-model clean validation scores.
- outputs/predictions/: 72 prediction CSVs, including stable sample IDs and clean/noisy text.
- outputs/test_metrics/: 13 metric records per run.
- outputs/summaries/all_test_metrics.csv: 936 evaluation records.
- outputs/summaries/three_seed_summary.csv: 312 groups with mean and sample std (ddof=1).
- outputs/summaries/training_summary.csv: actual/best epochs, stop reasons and run time.

Scores are on a 0-to-1 scale. Robustness drops are clean minus noisy scores
within each seed, then summarized across the three seeds.
IDs refer to zero-based rows of the frozen exported clean test CSV, not newly
invented original MASSIVE IDs. Test rows remain aligned across conditions.

## Software checks (no formal training)
```bash
cd /root/autodl-tmp/essentials_in_text_and_speech/02_training
CUDA_VISIBLE_DEVICES="" environment/bin/python -B -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES="" environment/bin/python -B -m src.check_data
CUDA_VISIBLE_DEVICES="" environment/bin/python -B -m src.check_environment
```
