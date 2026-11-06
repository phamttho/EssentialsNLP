"""Training implementations for the isolated, validation-selected tuning study.

No test data is read here. Temporary best weights are retained for the caller
to evaluate and verify before explicit, confined cleanup.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import warnings

import base


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_work(work):
    work = Path(work)
    root = base.ROOT.resolve()
    if work.is_symlink() or not work.resolve().is_relative_to(root) or work.resolve() == root:
        raise ValueError("Temporary model path must be strictly inside tuning root")
    work.mkdir(parents=True, exist_ok=True)
    for name in ("best.safetensors", "best.safetensors.tmp", "best.joblib", "best.joblib.tmp"):
        if (work / name).is_symlink():
            raise ValueError("Refusing symlink temporary model")
    return work


def chinese_tokens(text):
    import jieba
    path = base.ROOT / "cache" / "jieba"
    path.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(path)
    return [x for x in jieba.lcut(text) if x.strip()]


def _record_history(work, history):
    base.save_json(work / "training_history.json", history)


def fit_tfidf(lang, model_name, cond, seed, value, work):
    import numpy as np
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning

    old_cfg = json.loads((base.OLD / "configs" / "experiment.json").read_text())
    settings = dict(old_cfg[model_name])
    settings["ngram_range"] = tuple(settings["ngram_range"])
    settings["dtype"] = np.float64
    if model_name == "char_tfidf":
        settings["analyzer"] = "char"
    elif lang == "zh-CN":
        settings.update(tokenizer=chinese_tokens, token_pattern=None)
    lr = dict(old_cfg["logistic_regression"])
    lr["C"] = float(value)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**settings)),
        ("classifier", LogisticRegression(**lr, random_state=int(seed))),
    ])
    df = base.training(lang, cond)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(df.utterance.tolist(), df.intent.to_numpy())
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Logistic Regression did not converge; candidate is invalid")
    for w in caught:
        print("WARNING", str(w.message), flush=True)
    base.guard()
    path = work / "best.joblib"
    temp = work / "best.joblib.tmp"
    joblib.dump(pipeline, temp)
    temp.replace(path)
    restored = joblib.load(path)
    assert np.array_equal(
        pipeline.predict(df.utterance.iloc[:16].tolist()),
        restored.predict(df.utterance.iloc[:16].tolist()),
    ), "Reloaded pipeline differs"
    va = base.clean(lang, "validation")
    val = base.metrics(va.intent, restored.predict(va.utterance.tolist()))
    summary = {
        **val, "rows": len(df), "converged": True,
        "iterations": int(restored["classifier"].n_iter_.max()),
        "actual_epochs": None, "best_epoch": None,
        "stop_reason": "solver_converged",
        "weights_sha256": _sha(path), "weights_file": path.name,
        "parameter_name": "C", "parameter_value": float(value),
        "seed": int(seed), "best_validation_macro_f1": val["macro_f1"],
    }
    _record_history(work, [{"stage": "validation", **summary}])
    return restored.predict, summary


def make_best_callback(work, patience=10, threshold=0.0):
    from transformers import TrainerCallback

    class BestWeightsOnly(TrainerCallback):
        def __init__(self):
            self.best = None
            self.best_epoch = None
            self.best_step = None
            self.counter = 0
            self.active = True
            self.evaluations = []

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            if not self.active:
                return control
            from safetensors.torch import save_model
            score = float(metrics["eval_macro_f1"])
            if not math.isfinite(score):
                raise ValueError("Non-finite validation macro-F1")
            improved = self.best is None or score > self.best + threshold
            if improved:
                base.guard()
                path = work / "best.safetensors"
                tmp = work / "best.safetensors.tmp"
                # No optimizer, scheduler or RNG checkpoints are saved.
                save_model(model, str(tmp), force_contiguous=True)
                tmp.replace(path)
                self.best = score
                self.best_epoch = float(state.epoch)
                self.best_step = int(state.global_step)
                self.counter = 0
            else:
                self.counter += 1
            record = {
                "epoch": float(state.epoch), "global_step": int(state.global_step),
                "validation_macro_f1": score, "best_macro_f1": self.best,
                "best_epoch": self.best_epoch, "patience_counter": self.counter,
                "improved": improved,
            }
            self.evaluations.append(record)
            base.save_json(work / "validation_history.json", self.evaluations)
            _record_history(work, state.log_history)
            print("VALIDATION", json.dumps(record), flush=True)
            if self.counter >= patience:
                control.should_training_stop = True
            return control

    return BestWeightsOnly()


def _reload_best(model, work):
    from safetensors.torch import load_model
    missing, unexpected = load_model(model, str(work / "best.safetensors"), strict=True, device="cpu")
    assert not missing and not unexpected, (missing, unexpected)


def fit_xlmr(lang, cond, seed, value, work):
    import numpy as np
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, Trainer,
        TrainingArguments, DataCollatorWithPadding, set_seed,
    )

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2"}:
        raise RuntimeError("Exactly one allowed physical GPU (0, 1, 2) is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one usable CUDA device must be visible")
    old_cfg = json.loads((base.OLD / "configs" / "experiment.json").read_text())
    cfg = old_cfg["xlmr"]
    torch.set_num_threads(old_cfg["threads_per_worker"])
    set_seed(int(seed))
    source = base.snapshot()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels=None):
            self.tokens = tokenizer(texts, truncation=True, max_length=cfg["max_length"])
            self.labels = labels
        def __len__(self):
            return len(self.tokens["input_ids"])
        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.tokens.items()}
            if self.labels is not None:
                item["labels"] = int(self.labels[i])
            return item

    ids = base.label_mapping()
    model = AutoModelForSequenceClassification.from_pretrained(
        source, local_files_only=True, num_labels=60, id2label=ids,
        label2id={v: k for k, v in ids.items()}, attn_implementation="sdpa",
    )
    args = TrainingArguments(
        output_dir=str(work / "trainer"),
        num_train_epochs=cfg["max_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=float(value), weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"], warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="epoch", save_strategy="no",
        load_best_model_at_end=False, metric_for_best_model="macro_f1", greater_is_better=True,
        bf16=cfg["precision"] == "bf16", fp16=cfg["precision"] == "fp16",
        seed=int(seed), data_seed=int(seed), logging_steps=cfg["logging_steps"],
        logging_dir=str(work / "logs"), report_to=[], dataloader_num_workers=0,
        optim="adamw_torch", disable_tqdm=True, eval_accumulation_steps=8,
    )
    va = base.clean(lang, "validation")
    df = base.training(lang, cond)
    best = make_best_callback(work, int(cfg["patience"]), float(cfg["min_delta"]))
    trainer = Trainer(
        model=model, args=args,
        train_dataset=TextDataset(df.utterance.tolist(), df.intent.tolist()),
        eval_dataset=TextDataset(va.utterance.tolist(), va.intent.tolist()),
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
        compute_metrics=lambda x: base.metrics(x.label_ids, np.argmax(x.predictions, axis=-1)),
        callbacks=[best],
    )
    trainer.train()
    best.active = False
    _record_history(work, trainer.state.log_history)
    _reload_best(model, work)
    val = trainer.evaluate()
    if abs(float(val["eval_macro_f1"]) - best.best) >= 1e-9:
        raise RuntimeError("Reloaded best weights do not reproduce best validation macro-F1")
    actual = float(trainer.state.epoch)
    summary = {
        "accuracy": float(val["eval_accuracy"]), "macro_f1": float(val["eval_macro_f1"]),
        "rows": len(df), "actual_epochs": actual, "best_epoch": best.best_epoch,
        "global_step": int(trainer.state.global_step), "best_global_step": best.best_step,
        "best_validation_macro_f1": best.best,
        "patience_counter_at_stop": best.counter,
        "stop_reason": "early_stopping" if actual < cfg["max_epochs"] - 1e-6 else "epoch_limit",
        "weights_sha256": _sha(work / "best.safetensors"), "weights_file": "best.safetensors",
        "parameter_name": "learning_rate", "parameter_value": float(value), "seed": int(seed),
        "optimizer_checkpoints_saved": False, "resume_mode": "restart_interrupted_task",
    }
    def predict(texts):
        return np.argmax(trainer.predict(TextDataset(texts)).predictions, axis=-1)
    return predict, summary


def fit_job(lang, model, cond, seed, value, work):
    work = _checked_work(work)
    if model == "xlmr":
        return fit_xlmr(lang, cond, seed, value, work)
    if model not in {"word_tfidf", "char_tfidf"}:
        raise ValueError(model)
    return fit_tfidf(lang, model, cond, seed, value, work)


def smoke_test(work):
    """Tiny real CPU Trainer test of strict ties, patience, save and best reload."""
    work = _checked_work(work)
    import torch
    from transformers import RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments, set_seed
    set_seed(2028)
    torch.set_num_threads(1)
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, i):
            return {"input_ids": [0, 3 + i, 4, 2], "attention_mask": [1] * 4, "labels": i % 3}
    model = RobertaForSequenceClassification(RobertaConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=16, num_labels=3,
    ))
    callback = make_best_callback(work, patience=10)
    args = TrainingArguments(
        output_dir=str(work / "trainer"), num_train_epochs=30, use_cpu=True,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        eval_strategy="epoch", save_strategy="no", load_best_model_at_end=False,
        seed=2028, data_seed=2028, report_to=[], disable_tqdm=True, logging_strategy="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=TinyDataset(), eval_dataset=TinyDataset(),
                      compute_metrics=lambda x: {"macro_f1": 0.5}, callbacks=[callback])
    trainer.train()
    callback.active = False
    assert trainer.state.epoch == 11, trainer.state.epoch
    assert callback.best_epoch == 1 and callback.counter == 10
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _reload_best(model, work)
    assert any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items())
    assert trainer.evaluate()["eval_macro_f1"] == 0.5
    assert not list(work.rglob("checkpoint-*"))
    assert not list(work.rglob("optimizer.pt"))
    assert (work / "best.safetensors").is_file()
    base.save_json(work / "smoke_result.json", {
        "status": "PASS", "epochs": trainer.state.epoch, "best_epoch": callback.best_epoch,
        "patience_counter": callback.counter, "optimizer_checkpoint_count": 0,
        "test_data_accessed": False,
    })
    return {"status": "PASS", "epochs": 11, "best_epoch": 1, "patience_counter": 10}
