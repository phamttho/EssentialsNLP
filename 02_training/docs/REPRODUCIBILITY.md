# Reproducing and verifying the archived experiments

## Scope of this publication

This directory preserves the original experiment source and configuration bytes, plus retained metrics and evidence. The original 72-run study and the supplementary search are distinct experiments. Publication documentation is additional explanation; it is not a record of new training.

The code is **not a portable quick-start package**. It retains Linux-specific process handling, fixed deployment paths, offline model loading and checks that bind completed records to their original inputs. Reading the included JSON/CSV summaries needs no GPU. Full prediction verification or retraining requires the missing artifacts described below.

From `02_training/`, `python -B tools/verify_archive.py` checks the included archive against its source-file inventory and checks the recorded experiment/table counts. It uses only the Python standard library, without a GPU. This publication check does not retrain models or replace full verification of prediction files omitted from Git.

## Recorded execution layout

```text
/root/autodl-tmp/essentials_in_text_and_speech/
├── 01_data/                                  Git checkout of the frozen data source
│   └── exported_data/
│       ├── clean/
│       ├── augmented/
│       ├── corrupted_test/
│       └── metadata/
└── 02_training/                              Training directory within the checkout
    ├── environment/bin/python                Isolated Python environment
    ├── cache/pretrained/
    │   ├── metadata.json
    │   └── e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/
    ├── outputs/
    └── hyperparameter_tuning/
```

HPO `base.py` rejects any other experiment root. Launch scripts use absolute paths, and HPO `run.py` uses `02_training/environment/bin/python`. Original environment checks also require that environment prefix. The code uses `fcntl` and Linux `/proc`; Windows execution is not supported by the archived scripts.

Use a separate restoration machine or clean restoration directory. Do not replace a completed server experiment or reinterpret its archived completion records as a new run.

## Restore the data source

On a clean restoration machine, clone the repository into the project root. The checkout already contains `02_training/`. Clone the frozen data source separately into the sibling `01_data/` directory.

```bash
git clone --branch 02_training https://github.com/phamttho/EssentialsNLP.git \
  /root/autodl-tmp/essentials_in_text_and_speech
git clone --no-checkout https://github.com/phamttho/EssentialsNLP.git \
  /root/autodl-tmp/essentials_in_text_and_speech/01_data
git -C /root/autodl-tmp/essentials_in_text_and_speech/01_data \
  checkout --detach 6f6bd31ad98edc64bee375a203f516cb70efbcda
```

Record the exact publication commit you cloned; do not rely on a moving branch name for a later reproduction. The sibling `01_data` must remain a Git checkout because `src/check_data.py` reads its commit using `git rev-parse HEAD`.

Use the **frozen exported files**, not newly generated random corruptions. Verify their hashes against `outputs/manifests/data_check.json`. The archived report covers 66 files, including data metadata. `provenance/data_manifest.json` and `provenance/intent_mapping.json` describe the exports and the 60 intent labels; they do not replace the actual data files.

## Restore dependencies and pretrained files

The recorded environment used Python 3.12, PyTorch 2.8.0 with CUDA 12.8 support, Transformers 4.57.1, Accelerate 1.10.1, scikit-learn 1.7.2, pandas 2.3.3 and jieba 0.42.1. Consult `requirements.txt`, `requirements.lock.txt` and retained environment evidence for the complete recorded versions.

### Install and verify the locked environment

On Linux x86-64, use CPython 3.12 in a clean restoration checkout. The lock file pins the exact CUDA 12.8 PyTorch wheel and its SHA-256 checksum. Install from `requirements.lock.txt` to reproduce the recorded environment. `requirements.txt` is a shorter list of the main packages.

```bash
cd /root/autodl-tmp/essentials_in_text_and_speech/02_training
python3.12 -m venv environment
mkdir -p cache/pip tmp
PIP_CACHE_DIR="$PWD/cache/pip" TMPDIR="$PWD/tmp" \
  environment/bin/python -m pip install -r requirements.lock.txt
environment/bin/python -m pip check
environment/bin/python -B tools/verify_archive.py
```

Before running XLM-R on an RTX 5090, check that the restored environment can see the GPU and that PyTorch supports its CUDA architecture.

```bash
nvidia-smi
environment/bin/python -B - <<'PYTHON'
import torch

print(f"PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
assert torch.__version__ == "2.8.0+cu128", "Expected the PyTorch wheel from the lock file."
assert torch.cuda.is_available(), "A compatible NVIDIA GPU must be visible."
architectures = torch.cuda.get_arch_list()
print(f"Supported architectures: {', '.join(architectures)}")
assert "sm_120" in architectures, "The PyTorch build must support the RTX 5090."
PYTHON
```

These checks cover dependency consistency, the published archive and GPU support. Restore the frozen data and the pretrained files before launching the training scripts.

### Restore the pretrained snapshot

Restore `provenance/pretrained_metadata.json` **without editing its contents** to `02_training/cache/pretrained/metadata.json`. Obtain the recorded `FacebookAI/xlm-roberta-base` revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, then place these files in the snapshot directory shown above:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `sentencepiece.bpe.model`

Verify every file against the SHA-256 values in the metadata. The training code sets offline flags and will not download missing pretrained files automatically. The metadata contains the historical endpoint and absolute snapshot path; modifying that JSON also changes its fingerprint contribution.

## Restore omitted artifacts when needed

Complete original predictions remain at:

```text
/root/autodl-tmp/essentials_in_text_and_speech/02_training/outputs/predictions/<run_id>.csv
```

HPO validation and final predictions remain below:

```text
/root/autodl-tmp/essentials_in_text_and_speech/02_training/hyperparameter_tuning/search/<candidate>/
/root/autodl-tmp/essentials_in_text_and_speech/02_training/hyperparameter_tuning/results/final/<candidate>/
```

Request the matching files from Ke Sun or the existing private handoff. No public artifact-download link is supplied here. Restore the original filenames, compression formats and bytes, then check the recorded hashes; converting a file or recompressing it may change its file hash.

HPO reuses all 72 original validation records. For original parameter selections, it also verifies and imports original test predictions; the completed supplementary experiment reused 12 such runs. Original test metrics are needed for the baseline comparison. Original fine-tuned weights are not required for this reuse path.

The Git copy omits prediction files referenced by completed HPO records. Consequently, those records are evidence of the completed execution, **not a self-contained resume bundle**. Full `valid_record` checks will fail until the referenced files are restored. Relevant execution-state evidence is archived under `evidence/hyperparameter_tuning_state/`, not automatically installed as active process state.

## Fingerprints and a new run

The original fingerprint includes source, configuration, input hashes, dependency-lock contents and pretrained metadata. The HPO fingerprint additionally includes all top-level HPO Python files, the plan, the original data-check report and the 72 original validation files. Preserve line endings and file bytes when verifying an archived run; see `provenance/source_files.json` for the published source inventory.

Do not edit code/configuration and then reuse old completion records as though they belonged to the modified experiment. A portable implementation, different GPU schedule or changed training setting is a new version and must have separate fresh outputs and fingerprints.

For a fresh rerun, separate the bundled evidence from active outputs before starting, and prepare the required original-stage evidence before HPO. For an exact archive restoration, restore every referenced artifact before attempting resume or verification. These are different workflows; both require restoring the runtime assets.

Only after restoring and checking all prerequisites should the recorded launch scripts be used. The original `scripts/start.sh` schedules four GPUs (0–3); `hyperparameter_tuning/scripts/start.sh` schedules three GPUs (0–2). Check `nvidia-smi` and existing processes before either launch. The archived original configuration is not suitable for a three-GPU machine without a separately versioned scheduler change.

HPO initially requires at least 36.5 GiB free at full configured concurrency. That threshold is not an estimate for storing an entire fresh original experiment: the original stage retains model weights and has additional disk requirements. New HPO weights are removed only after prediction verification. Interrupted HPO fits restart from scratch because optimizer recovery checkpoints were not retained.

## Scientific interpretation

Selection uses the mean clean-validation macro-F1 over two training conditions and three seeds for each language/model pair. The 216 candidate records comprise 72 reused and 144 new results. The final 72 evaluations comprise 60 refits and 12 reused runs. Do not describe all 216 records as new training, or choose the best seed when reporting results.

The frozen data cleaning, held-out character-pool use and synthetic-typo limitations remain unchanged by HPO. Original test results had already been viewed, so describe the tuning as supplementary. Full prediction checks establish consistency with saved predictions and labels; they do not establish realistic typo coverage or guarantee improvements.

Training histories retain XLM-R training loss at logging steps and validation loss, accuracy and macro-F1 at epoch evaluations. They do not include per-epoch training accuracy. W&B or another plotting tool can visualize retained values later, but cannot recover metrics that were never measured. Making predictions on new inputs with a refitted tuned model requires retraining. The 24 original XLM-R weights were retired after the prediction backup was verified. Original checks that require those weights need a separate weight copy or a fresh fit, while analysis from saved predictions can proceed.
