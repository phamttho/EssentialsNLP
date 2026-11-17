"""Test exports for selected candidates only; no parameter selection uses this module."""
from pathlib import Path
import itertools
import json
import math
import gzip
import os
import time
import numpy as np
import pandas as pd
from base import ROOT, OLD, DATA, clean, test_sets, metrics, label_mapping, save_json, sha

LANGUAGES = ("en-US", "vi-VN", "de-DE", "zh-CN")
MODELS = ("word_tfidf", "char_tfidf", "xlmr")
CONDITIONS = ("clean", "mixed")
SEEDS = (2028, 2029, 2030)
EXPECTED_TEST_ROWS = 2974
EXPECTED_CONDITIONS = 13
IDENTITY = ["language", "model", "training_condition", "seed", "test_condition", "source_row"]
BASE_COLUMNS = ["source_row", "sample_id", "true_intent", "predicted_intent",
                "true_intent_name", "predicted_intent_name", "clean_utterance",
                "test_utterance", "language", "model", "training_condition",
                "test_condition", "typo_type", "noise_level", "seed"]
MEASURES = ["accuracy", "macro_f1", "accuracy_drop", "macro_f1_drop"]


def _destination(dest):
    dest = Path(dest).resolve()
    root = ROOT.resolve()
    if dest == root or not dest.is_relative_to(root):
        raise ValueError("Export destination must be a child of the new tuning root")
    return dest


def _info(lang, model, cond, seed, value):
    if lang not in LANGUAGES or model not in MODELS or cond not in CONDITIONS or seed not in SEEDS:
        raise ValueError("Unknown final evaluation identity")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("Invalid hyperparameter")
    return {"language": lang, "model": model, "training_condition": cond,
            "seed": int(seed), "parameter_name": "learning_rate" if model == "xlmr" else "C",
            "parameter_value": float(value)}


def _sets(lang):
    expected = list(test_sets(lang))
    assert len(expected) == EXPECTED_CONDITIONS, "Expected 13 frozen test conditions"
    names = [name for name, _, _, _ in expected]
    assert names[0] == "clean" and len(set(names)) == EXPECTED_CONDITIONS
    expected_names = {"clean"} | {f"{typo}_{level:.2f}" for typo in
                                 ("deletion", "substitution", "duplication", "swap")
                                 for level in (.05, .10, .20)}
    assert set(names) == expected_names, "Unexpected test condition grid"
    for name, typo, level, df in expected:
        assert len(df) == EXPECTED_TEST_ROWS, name
        assert not df.source_row.duplicated().any(), name
        assert set(df.source_row) == set(range(EXPECTED_TEST_ROWS)), name
        assert set(df.intent.astype(int)) <= set(label_mapping()), name
    return expected


def _write_predictions(frame, path):
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8",
                 compression={"method": "gzip", "compresslevel": 6, "mtime": 0})
    os.replace(tmp, path)


def _close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= 1e-12


def verify_existing(lang, model, cond, seed, value, dest, check_marker=True):
    """Reload saved CSV and recompute every metric against frozen source data."""
    dest = _destination(dest)
    info = _info(lang, model, cond, seed, value)
    pp, mp = dest/"predictions.csv.gz", dest/"metrics.json"
    frame = pd.read_csv(pp, keep_default_na=False, compression="gzip")
    stored = json.loads(mp.read_text(encoding="utf-8"))
    assert set(BASE_COLUMNS + ["parameter_name", "parameter_value"]) <= set(frame.columns)
    assert len(frame) == EXPECTED_TEST_ROWS * EXPECTED_CONDITIONS
    assert not frame.duplicated(IDENTITY).any(), "Duplicate prediction keys"
    assert len(stored) == EXPECTED_CONDITIONS
    for k, v in info.items():
        assert (frame[k] == v).all(), f"Incorrect field {k}"
    ids = label_mapping()
    assert set(ids) == set(range(60))
    for name in ["source_row", "true_intent", "predicted_intent"]:
        numeric = pd.to_numeric(frame[name], errors="raise")
        assert np.isfinite(numeric).all()
        assert (numeric == numeric.astype(np.int64)).all()
        frame[name] = numeric.astype(np.int64)
    assert set(frame.true_intent) <= set(ids)
    assert set(frame.predicted_intent) <= set(ids)
    by_name = {r["test_condition"]: r for r in stored}
    assert len(by_name) == EXPECTED_CONDITIONS
    expected = _sets(lang)
    assert set(frame.test_condition) == set(by_name) == {s[0] for s in expected}
    clean_metric = None
    for name, typo, level, df in expected:
        actual = frame.loc[frame.test_condition == name].sort_values("source_row")
        source = df.sort_values("source_row")
        assert len(actual) == len(source) == EXPECTED_TEST_ROWS
        assert np.array_equal(actual.source_row.to_numpy(), source.source_row.to_numpy())
        assert np.array_equal(actual.true_intent.to_numpy(), source.intent.to_numpy())
        assert actual.sample_id.tolist() == [f"{lang}:test:{i}" for i in source.source_row]
        assert actual.clean_utterance.tolist() == source.clean_utterance.tolist()
        assert actual.test_utterance.tolist() == source.utterance.tolist()
        assert actual.true_intent_name.tolist() == [ids[int(i)] for i in source.intent]
        assert actual.predicted_intent_name.tolist() == [ids[int(i)] for i in actual.predicted_intent]
        assert (actual.typo_type == typo).all() and (actual.noise_level == float(level)).all()
        recomputed = metrics(actual.true_intent, actual.predicted_intent)
        if name == "clean":
            clean_metric = recomputed
        assert clean_metric is not None
        recomputed.update(accuracy_drop=clean_metric["accuracy"]-recomputed["accuracy"],
                          macro_f1_drop=clean_metric["macro_f1"]-recomputed["macro_f1"])
        row = by_name[name]
        for k, v in {**info, "typo_type": typo, "noise_level": float(level),
                     "n_samples": len(source)}.items():
            assert row[k] == v, f"Metric metadata mismatch {name}: {k}"
        for k in MEASURES:
            assert _close(row[k], recomputed[k]), f"Metric mismatch {name}: {k}"
    result = {**info, "verified": True, "prediction_rows": len(frame),
              "test_conditions": len(stored), "predictions_sha256": sha(pp),
              "metrics_sha256": sha(mp)}
    marker = dest/"verification.json"
    if check_marker:
        assert marker.is_file(), "Missing saved verification marker"
        previous = json.loads(marker.read_text(encoding="utf-8"))
        for k, v in result.items():
            assert previous[k] == v, f"Verification marker mismatch: {k}"
    return result


def _finish(lang, model, cond, seed, value, dest, origin):
    result = verify_existing(lang, model, cond, seed, value, dest, check_marker=False)
    save_json(dest/"verification.json", {**result, "origin": origin, "verified_at": time.time()})
    return result


def export_and_verify(lang, model, cond, seed, value, predict, dest):
    """Produce selected-model test outputs; caller deletes weights only on success."""
    dest = _destination(dest)
    dest.mkdir(parents=True, exist_ok=True)
    info = _info(lang, model, cond, seed, value)
    ids = label_mapping()
    frames, rows = [], []
    for name, typo, level, df in _sets(lang):
        raw = np.asarray(predict(df.utterance.tolist()))
        assert raw.ndim == 1 and len(raw) == len(df)
        assert np.isfinite(raw).all() and np.array_equal(raw, raw.astype(np.int64))
        pred = raw.astype(np.int64)
        assert set(pred) <= set(ids)
        extra = {**info, "test_condition": name, "typo_type": typo, "noise_level": float(level)}
        out = pd.DataFrame({
            "source_row": df.source_row.to_numpy(),
            "sample_id": [f"{lang}:test:{i}" for i in df.source_row],
            "true_intent": df.intent.to_numpy(), "predicted_intent": pred,
            "true_intent_name": [ids[int(i)] for i in df.intent],
            "predicted_intent_name": [ids[int(i)] for i in pred],
            "clean_utterance": df.clean_utterance.to_numpy(),
            "test_utterance": df.utterance.to_numpy()})
        for k, v in extra.items(): out[k] = v
        frames.append(out)
        row = {**extra, "n_samples": len(df), **metrics(df.intent, pred)}
        rows.append(row)
        print("TEST", lang, model, cond, seed, name, row["macro_f1"], flush=True)
    for row in rows:
        row["accuracy_drop"] = rows[0]["accuracy"] - row["accuracy"]
        row["macro_f1_drop"] = rows[0]["macro_f1"] - row["macro_f1"]
    _write_predictions(pd.concat(frames, ignore_index=True), dest/"predictions.csv.gz")
    save_json(dest/"metrics.json", rows)
    return _finish(lang, model, cond, seed, value, dest, "new_selected_model")


def import_legacy(lang, model, cond, seed, value, dest):
    """Reuse an original candidate only after checking its recorded file hashes."""
    default = 2e-5 if model == "xlmr" else 1.0
    assert float(value) == default, "Only the original parameter can reuse legacy predictions"
    info = _info(lang, model, cond, seed, value)
    rid = f"{lang}__{model}__{cond}__seed{seed}"
    old = OLD.resolve()
    manifest = json.loads((old/"outputs/manifests"/f"{rid}.json").read_text())
    assert manifest["status"] == "completed" and manifest["run_id"] == rid
    config_value = (manifest["config"]["xlmr"]["learning_rate"] if model == "xlmr"
                    else manifest["config"]["logistic_regression"]["C"])
    assert float(config_value) == default
    pp = old/"outputs/predictions"/f"{rid}.csv"
    mp = old/"outputs/test_metrics"/f"{rid}.json"
    assert sha(pp) == manifest["predictions_sha256"], "Legacy prediction hash mismatch"
    assert sha(mp) == manifest["metrics_sha256"], "Legacy metric hash mismatch"
    frame = pd.read_csv(pp, keep_default_na=False)
    for k in ["parameter_name", "parameter_value"]: frame[k] = info[k]
    rows = json.loads(mp.read_text())
    for row in rows:
        row["parameter_name"] = info["parameter_name"]
        row["parameter_value"] = info["parameter_value"]
    dest = _destination(dest); dest.mkdir(parents=True, exist_ok=True)
    _write_predictions(frame, dest/"predictions.csv.gz")
    save_json(dest/"metrics.json", rows)
    result = _finish(lang, model, cond, seed, value, dest, "verified_original_run")
    save_json(dest/"legacy_source.json", {
        "run_id": rid, "legacy_predictions_sha256": manifest["predictions_sha256"],
        "legacy_metrics_sha256": manifest["metrics_sha256"],
        "legacy_manifest_sha256": sha(old/"outputs/manifests"/f"{rid}.json"),
        "legacy_fingerprint": manifest["fingerprint"]})
    return result


def _training_summary(entries):
    rows = []
    for e in entries:
        lang,model,cond,seed,value = e["language"],e["model"],e["condition"],e["seed"],e["value"]
        rid = f"{lang}__{model}__{cond}__seed{seed}"
        cid = rid + "__value" + format(value, ".8g")
        rec = json.loads((ROOT/"records/final"/(cid+".json")).read_text())
        assert rec["status"] == "completed"
        assert rec["job"] == [lang,model,cond,seed,value]
        val = rec["summary"]
        if rec["provenance"] == "original_run_reused":
            old = json.loads((OLD/"outputs/manifests"/(rid+".json")).read_text())
            training = old["training_summary"]
        else:
            training = val
        row = {"language":lang,"model":model,"training_condition":cond,"seed":seed,
               "parameter_name":"learning_rate" if model=="xlmr" else "C",
               "parameter_value":value,"provenance":rec["provenance"],
               "validation_accuracy":val["accuracy"],"validation_macro_f1":val["macro_f1"],
               "search_validation_macro_f1":rec.get("search_macro_f1",val["macro_f1"]),
               "refit_validation_difference":rec.get("refit_validation_difference",0.0),
               "final_stage_wall_minutes":(rec["finished"]-rec["started"])/60}
        for field in ("rows","actual_epochs","best_epoch","global_step","iterations",
                      "converged","stop_reason","patience_counter_at_stop"):
            row[field] = training.get(field)
        rows.append(row)
    return pd.DataFrame(rows)


def _baseline_comparison(frame,entries):
    keys = ["language","model","training_condition","test_condition","typo_type","noise_level"]
    originals = []
    for e in entries:
        rid = f'{e["language"]}__{e["model"]}__{e["condition"]}__seed{e["seed"]}'
        manifest = json.loads((OLD/"outputs/manifests"/(rid+".json")).read_text())
        mp = OLD/"outputs/test_metrics"/(rid+".json")
        assert manifest["status"]=="completed" and sha(mp)==manifest["metrics_sha256"]
        originals.extend(json.loads(mp.read_text()))
    original = pd.DataFrame(originals)
    assert len(original)==936
    join = keys+["seed"]
    assert not original.duplicated(join).any()
    merged = frame.merge(original[join+MEASURES],on=join,how="inner",
                         suffixes=("_tuned","_original"),validate="one_to_one")
    assert len(merged)==936
    measures = []
    for metric in MEASURES:
        delta=metric+"_delta"
        merged[delta]=merged[metric+"_tuned"]-merged[metric+"_original"]
        measures.extend([metric+"_original",metric+"_tuned",delta])
    groupkeys=keys+["parameter_name","parameter_value"]
    for _,group in merged.groupby(groupkeys,dropna=False):
        assert sorted(group.seed.tolist())==list(SEEDS)
    result=merged.groupby(groupkeys,dropna=False)[measures].agg(["mean","std"])
    result.columns=["_".join(c) for c in result.columns]
    result=result.reset_index()
    result["n_seeds"]=3
    result["original_parameter_value"]=np.where(result.model=="xlmr",2e-5,1.0)
    assert len(result)==312
    return result


SUMMARY_NOTES = """Final selected-parameter analysis package
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
"""

def aggregate(entries):
    """entries: list[{language, model, condition, seed, value, dest}], exactly 72."""
    expected = set(itertools.product(LANGUAGES, MODELS, CONDITIONS, SEEDS))
    identities = [(e["language"], e["model"], e["condition"], e["seed"]) for e in entries]
    assert len(entries) == 72 and set(identities) == expected and len(set(identities)) == 72
    parts = []
    for e in entries:
        verify_existing(e["language"],e["model"],e["condition"],e["seed"],e["value"],e["dest"])
        parts.extend(json.loads((_destination(e["dest"])/"metrics.json").read_text()))
    frame = pd.DataFrame(parts)
    assert len(frame) == 936
    # The selected parameter is shared by both conditions and all seeds per language/model.
    assert (frame.groupby(["language","model"]).parameter_value.nunique() == 1).all()
    keys = ["language","model","training_condition","test_condition","typo_type",
            "noise_level","parameter_name","parameter_value"]
    for _, group in frame.groupby(keys, dropna=False):
        assert sorted(group.seed.tolist()) == list(SEEDS)
    grouped = frame.groupby(keys, dropna=False)[MEASURES].agg(["mean","std"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    summary = grouped.reset_index(); summary["n_seeds"] = 3
    assert len(summary) == 312
    folder = _destination(ROOT/"results/summaries"); folder.mkdir(parents=True, exist_ok=True)
    training_summary=_training_summary(entries)
    comparison=_baseline_comparison(frame,entries)
    for name, data in [("all_test_metrics.csv",frame),("three_seed_summary.csv",summary),
                       ("final_training_summary.csv",training_summary),("baseline_vs_tuned.csv",comparison)]:
        temp = folder/(name+".tmp")
        data.to_csv(temp,index=False); os.replace(temp,folder/name)
    notes=folder/"README.txt"
    notes_tmp=folder/"README.txt.tmp";notes_tmp.write_text(SUMMARY_NOTES,encoding="utf-8");os.replace(notes_tmp,notes)
    result = {"status":"completed","runs":72,"evaluation_rows":936,"summary_rows":312,
              "seeds":list(SEEDS),"std_ddof":1,"metric_scale":"0 to 1",
              "drop":"clean minus noisy within seed, then mean/std",
              "all_test_metrics_sha256":sha(folder/"all_test_metrics.csv"),
              "three_seed_summary_sha256":sha(folder/"three_seed_summary.csv"),
              "final_training_summary_sha256":sha(folder/"final_training_summary.csv"),
              "baseline_vs_tuned_sha256":sha(folder/"baseline_vs_tuned.csv")}
    save_json(folder/"verification.json",result)
    return result
