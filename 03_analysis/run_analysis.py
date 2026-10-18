from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

LANG_ORDER = ["en-US", "vi-VN", "de-DE", "zh-CN"]
MODEL_ORDER = ["word_tfidf", "char_tfidf", "xlmr"]
TRAIN_ORDER = ["clean", "mixed"]
TYPO_ORDER = ["deletion", "substitution", "duplication", "swap"]
NOISE_ORDER = [0.05, 0.10, 0.20]
MODEL_LABEL = {
    "word_tfidf": "Word TF-IDF + LR",
    "char_tfidf": "Char TF-IDF + LR",
    "xlmr": "XLM-R-base",
}
LANG_LABEL = {
    "en-US": "English",
    "vi-VN": "Vietnamese",
    "de-DE": "German",
    "zh-CN": "Simplified Chinese",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_inputs(handoff: Path):
    summaries = handoff / "outputs" / "summaries"
    all_metrics = pd.read_csv(summaries / "all_test_metrics.csv")
    three_seed = pd.read_csv(summaries / "three_seed_summary.csv")
    training = pd.read_csv(summaries / "training_summary.csv")
    with (handoff / "data_metadata" / "intent_mapping.json").open(encoding="utf-8") as f:
        intent_mapping = json.load(f)
    return all_metrics, three_seed, training, intent_mapping


def integrity_checks(handoff: Path, all_metrics: pd.DataFrame, three_seed: pd.DataFrame) -> dict:
    pred_files = sorted((handoff / "outputs" / "predictions").glob("*.csv"))
    run_keys = all_metrics[["language", "model", "training_condition", "seed"]].drop_duplicates()
    cond_counts = all_metrics.groupby(["language", "model", "training_condition", "seed"]).size()
    checks = {
        "prediction_files": len(pred_files),
        "run_groups": len(run_keys),
        "test_metric_rows": len(all_metrics),
        "three_seed_summary_rows": len(three_seed),
        "languages": sorted(all_metrics.language.unique().tolist()),
        "models": sorted(all_metrics.model.unique().tolist()),
        "training_conditions": sorted(all_metrics.training_condition.unique().tolist()),
        "seeds": sorted(all_metrics.seed.unique().tolist()),
        "min_test_conditions_per_run": int(cond_counts.min()),
        "max_test_conditions_per_run": int(cond_counts.max()),
    }
    assert checks["prediction_files"] == 72, checks
    assert checks["run_groups"] == 72, checks
    assert checks["test_metric_rows"] == 936, checks
    assert checks["three_seed_summary_rows"] == 312, checks
    assert checks["min_test_conditions_per_run"] == 13 and checks["max_test_conditions_per_run"] == 13, checks
    return checks


def per_seed_clean_noisy(all_metrics: pd.DataFrame) -> pd.DataFrame:
    clean = all_metrics[all_metrics.test_condition == "clean"][[
        "language", "model", "training_condition", "seed", "accuracy", "macro_f1"
    ]].rename(columns={"accuracy": "clean_accuracy", "macro_f1": "clean_macro_f1"})
    noisy = (
        all_metrics[all_metrics.test_condition != "clean"]
        .groupby(["language", "model", "training_condition", "seed"], as_index=False)
        .agg(noisy_accuracy=("accuracy", "mean"), noisy_macro_f1=("macro_f1", "mean"))
    )
    out = clean.merge(noisy, on=["language", "model", "training_condition", "seed"], validate="one_to_one")
    out["accuracy_drop"] = out.clean_accuracy - out.noisy_accuracy
    out["macro_f1_drop"] = out.clean_macro_f1 - out.noisy_macro_f1
    return out


def make_tables(all_metrics: pd.DataFrame, three_seed: pd.DataFrame, out_dir: Path) -> dict[str, pd.DataFrame]:
    tables = ensure_dir(out_dir / "tables")
    per_seed = per_seed_clean_noisy(all_metrics)

    by_language = (
        per_seed.groupby(["language", "model", "training_condition"], as_index=False)
        .agg(
            clean_f1_mean=("clean_macro_f1", "mean"),
            clean_f1_std=("clean_macro_f1", "std"),
            noisy_f1_mean=("noisy_macro_f1", "mean"),
            noisy_f1_std=("noisy_macro_f1", "std"),
            robustness_drop_mean=("macro_f1_drop", "mean"),
            robustness_drop_std=("macro_f1_drop", "std"),
        )
    )
    for c in [c for c in by_language.columns if c.endswith("mean") or c.endswith("std")]:
        by_language[c] *= 100
    by_language["model_label"] = by_language.model.map(MODEL_LABEL)
    by_language["language_label"] = by_language.language.map(LANG_LABEL)
    by_language.to_csv(tables / "01_by_language.csv", index=False)

    overall = (
        by_language.groupby(["model", "training_condition"], as_index=False)
        .agg(
            clean_f1=("clean_f1_mean", "mean"),
            noisy_f1=("noisy_f1_mean", "mean"),
            robustness_drop=("robustness_drop_mean", "mean"),
        )
    )
    overall["model_label"] = overall.model.map(MODEL_LABEL)
    overall["model"] = pd.Categorical(overall.model, MODEL_ORDER, ordered=True)
    overall["training_condition"] = pd.Categorical(overall.training_condition, TRAIN_ORDER, ordered=True)
    overall = overall.sort_values(["model", "training_condition"]).reset_index(drop=True)
    overall.to_csv(tables / "02_overall_equal_language_weight.csv", index=False)

    base = by_language[(by_language.model == "word_tfidf") & (by_language.training_condition == "clean")][
        ["language", "clean_f1_mean", "noisy_f1_mean", "robustness_drop_mean"]
    ].rename(columns={
        "clean_f1_mean": "baseline_clean_f1",
        "noisy_f1_mean": "baseline_noisy_f1",
        "robustness_drop_mean": "baseline_drop",
    })
    comp = by_language[by_language.training_condition == "clean"].merge(base, on="language", validate="many_to_one")
    comp["clean_f1_vs_baseline"] = comp.clean_f1_mean - comp.baseline_clean_f1
    comp["noisy_f1_vs_baseline"] = comp.noisy_f1_mean - comp.baseline_noisy_f1
    comp["drop_vs_baseline"] = comp.robustness_drop_mean - comp.baseline_drop
    comp.to_csv(tables / "03_clean_model_baseline_comparison.csv", index=False)

    wide = by_language.pivot(index=["language", "model"], columns="training_condition", values=["clean_f1_mean", "noisy_f1_mean", "robustness_drop_mean"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    aug = wide.reset_index()
    aug["clean_f1_change_mixed_minus_clean"] = aug.clean_f1_mean_mixed - aug.clean_f1_mean_clean
    aug["noisy_f1_gain_mixed_minus_clean"] = aug.noisy_f1_mean_mixed - aug.noisy_f1_mean_clean
    aug["drop_reduction_clean_minus_mixed"] = aug.robustness_drop_mean_clean - aug.robustness_drop_mean_mixed
    aug["model_label"] = aug.model.map(MODEL_LABEL)
    aug["language_label"] = aug.language.map(LANG_LABEL)
    aug.to_csv(tables / "04_augmentation_effect.csv", index=False)

    noisy = all_metrics[all_metrics.test_condition != "clean"].copy()
    by_type_seed = noisy.groupby(["language", "model", "training_condition", "seed", "typo_type"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), macro_f1_drop=("macro_f1_drop", "mean")
    )
    by_type = by_type_seed.groupby(["language", "model", "training_condition", "typo_type"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        drop_mean=("macro_f1_drop", "mean"), drop_std=("macro_f1_drop", "std")
    )
    for c in ["macro_f1_mean", "macro_f1_std", "drop_mean", "drop_std"]:
        by_type[c] *= 100
    by_type.to_csv(tables / "05_by_typo_type.csv", index=False)

    by_level_seed = noisy.groupby(["language", "model", "training_condition", "seed", "noise_level"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"), macro_f1_drop=("macro_f1_drop", "mean")
    )
    by_level = by_level_seed.groupby(["language", "model", "training_condition", "noise_level"], as_index=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        drop_mean=("macro_f1_drop", "mean"), drop_std=("macro_f1_drop", "std")
    )
    for c in ["macro_f1_mean", "macro_f1_std", "drop_mean", "drop_std"]:
        by_level[c] *= 100
    by_level.to_csv(tables / "06_by_noise_level.csv", index=False)

    noisy_summary = three_seed[three_seed.test_condition != "clean"].copy()
    worst_idx = noisy_summary.groupby(["language", "model", "training_condition"])["macro_f1_mean"].idxmin()
    worst = noisy_summary.loc[worst_idx, [
        "language", "model", "training_condition", "test_condition", "typo_type", "noise_level",
        "macro_f1_mean", "macro_f1_std", "macro_f1_drop_mean", "macro_f1_drop_std"
    ]].copy()
    for c in ["macro_f1_mean", "macro_f1_std", "macro_f1_drop_mean", "macro_f1_drop_std"]:
        worst[c] *= 100
    worst.to_csv(tables / "07_worst_condition_per_system.csv", index=False)

    return {
        "per_seed": per_seed,
        "by_language": by_language,
        "overall": overall,
        "baseline": comp,
        "augmentation": aug,
        "by_type": by_type,
        "by_level": by_level,
        "worst": worst,
    }


def plot_overall(overall: pd.DataFrame, fig_dir: Path):
    df = overall.copy()
    df["label"] = df.apply(lambda r: f"{MODEL_LABEL[r.model]}\n{r.training_condition}", axis=1)
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, df.clean_f1, width, label="Clean test")
    ax.bar(x + width/2, df.noisy_f1, width, label="Mean over 12 corrupted tests")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Clean vs corrupted-test performance")
    ax.set_xticks(x, df.label, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "01_clean_vs_noisy_macro_f1.png", dpi=200)
    plt.close(fig)


def plot_augmentation(aug: pd.DataFrame, fig_dir: Path):
    df = aug.copy().sort_values(["model", "language"])
    df["label"] = df.apply(lambda r: f"{LANG_LABEL[r.language]}\n{MODEL_LABEL[r.model]}", axis=1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(df)), df.noisy_f1_gain_mixed_minus_clean)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Noisy macro-F1 gain from mixed training (points)")
    ax.set_title("Effect of typo-augmented training on corrupted inputs")
    ax.set_xticks(np.arange(len(df)), df.label, rotation=55, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_augmentation_noisy_gain.png", dpi=200)
    plt.close(fig)


def plot_noise_curves(by_level: pd.DataFrame, fig_dir: Path):
    df = by_level[(by_level.model == "xlmr") & (by_level.training_condition == "mixed")]
    fig, ax = plt.subplots(figsize=(8, 6))
    for lang in LANG_ORDER:
        g = df[df.language == lang].sort_values("noise_level")
        ax.plot(g.noise_level * 100, g.macro_f1_mean, marker="o", label=LANG_LABEL[lang])
    ax.set_xlabel("Nominal corruption level (%)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("XLM-R mixed training: performance as noise increases")
    ax.set_xticks([5, 10, 20])
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "03_xlmr_mixed_noise_curve.png", dpi=200)
    plt.close(fig)


def plot_typo_type(by_type: pd.DataFrame, fig_dir: Path):
    df = by_type[(by_type.model == "xlmr") & (by_type.training_condition == "mixed")]
    agg = df.groupby("typo_type", as_index=False).drop_mean.mean()
    agg["typo_type"] = pd.Categorical(agg.typo_type, TYPO_ORDER, ordered=True)
    agg = agg.sort_values("typo_type")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(agg.typo_type.astype(str), agg.drop_mean)
    ax.set_ylabel("Mean macro-F1 drop (points)")
    ax.set_title("XLM-R mixed training: average degradation by corruption type")
    fig.tight_layout()
    fig.savefig(fig_dir / "04_xlmr_mixed_drop_by_typo_type.png", dpi=200)
    plt.close(fig)


def plot_language_robustness(by_language: pd.DataFrame, fig_dir: Path):
    df = by_language[(by_language.model == "xlmr") & (by_language.training_condition == "mixed")].copy()
    df["language"] = pd.Categorical(df.language, LANG_ORDER, ordered=True)
    df = df.sort_values("language")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df.language.map(LANG_LABEL).astype(str), df.robustness_drop_mean)
    ax.set_ylabel("Clean-to-noisy macro-F1 drop (points)")
    ax.set_title("XLM-R mixed training: robustness drop by language")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(fig_dir / "05_xlmr_mixed_robustness_by_language.png", dpi=200)
    plt.close(fig)


def plot_baseline_comparison(comp: pd.DataFrame, fig_dir: Path):
    df = comp.copy()
    # Equal-weight languages within each model, clean training only.
    agg = df.groupby("model", as_index=False).agg(noisy_f1=("noisy_f1_mean", "mean"))
    agg["model"] = pd.Categorical(agg.model, MODEL_ORDER, ordered=True)
    agg = agg.sort_values("model")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(agg.model.map(MODEL_LABEL).astype(str), agg.noisy_f1)
    ax.set_ylabel("Mean corrupted-test macro-F1 (%)")
    ax.set_title("Clean-trained models: robustness comparison")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_clean_trained_model_comparison.png", dpi=200)
    plt.close(fig)


def majority_vote(values):
    vals = [int(v) for v in values if pd.notna(v)]
    if not vals:
        return np.nan, 0
    counts = Counter(vals)
    label, count = counts.most_common(1)[0]
    # With three seeds, a strict majority means >=2. Do not force consensus for 1/1/1.
    if count < 2:
        return np.nan, count
    return label, count


def load_xlmr_predictions(handoff: Path, language: str, training_condition: str) -> pd.DataFrame:
    frames = []
    for seed in [2028, 2029, 2030]:
        p = handoff / "outputs" / "predictions" / f"{language}__xlmr__{training_condition}__seed{seed}.csv"
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def aggregate_majority(pred: pd.DataFrame) -> pd.DataFrame:
    keys = ["language", "training_condition", "test_condition", "source_row"]
    fixed = pred.groupby(keys, as_index=False).agg(
        true_intent=("true_intent", "first"),
        true_intent_name=("true_intent_name", "first"),
        clean_utterance=("clean_utterance", "first"),
        test_utterance=("test_utterance", "first"),
        typo_type=("typo_type", "first"),
        noise_level=("noise_level", "first"),
    )
    votes = (
        pred.groupby(keys)["predicted_intent"]
        .apply(lambda s: majority_vote(s.tolist()))
        .reset_index(name="vote")
    )
    votes[["majority_prediction", "agreement_count"]] = pd.DataFrame(votes.vote.tolist(), index=votes.index)
    votes = votes.drop(columns="vote")
    names = pred[["predicted_intent", "predicted_intent_name"]].drop_duplicates().set_index("predicted_intent")["predicted_intent_name"].to_dict()
    out = fixed.merge(votes, on=keys, validate="one_to_one")
    out["majority_prediction_name"] = out.majority_prediction.map(names)
    out["majority_correct"] = out.majority_prediction.eq(out.true_intent)
    return out


def error_analysis(handoff: Path, tables_dir: Path, worst: pd.DataFrame):
    all_errors = []
    all_confusions = []
    rescue_examples = []

    for lang in LANG_ORDER:
        mixed_raw = load_xlmr_predictions(handoff, lang, "mixed")
        clean_raw = load_xlmr_predictions(handoff, lang, "clean")
        mixed = aggregate_majority(mixed_raw)
        clean_train = aggregate_majority(clean_raw)

        worst_mixed = worst[(worst.language == lang) & (worst.model == "xlmr") & (worst.training_condition == "mixed")].iloc[0]
        condition = worst_mixed.test_condition
        clean_rows = mixed[mixed.test_condition == "clean"][
            ["source_row", "majority_prediction", "agreement_count", "majority_correct"]
        ].rename(columns={
            "majority_prediction": "clean_majority_prediction",
            "agreement_count": "clean_agreement_count",
            "majority_correct": "clean_majority_correct",
        })
        noisy_rows = mixed[mixed.test_condition == condition].merge(clean_rows, on="source_row", validate="one_to_one")
        failures = noisy_rows[
            noisy_rows.clean_majority_correct & (~noisy_rows.majority_correct) & noisy_rows.majority_prediction.notna()
        ].copy()
        failures["priority"] = (failures.clean_agreement_count == 3).astype(int) + (failures.agreement_count == 3).astype(int)
        failures = failures.sort_values(["priority", "agreement_count"], ascending=False).head(8)
        failures["selected_worst_condition"] = condition
        all_errors.append(failures[[
            "language", "source_row", "selected_worst_condition", "typo_type", "noise_level",
            "clean_utterance", "test_utterance", "true_intent_name", "majority_prediction_name",
            "clean_agreement_count", "agreement_count"
        ]])

        conf = noisy_rows[noisy_rows.majority_prediction.notna() & (~noisy_rows.majority_correct)].groupby(
            ["true_intent_name", "majority_prediction_name"], as_index=False
        ).size().sort_values("size", ascending=False).head(10)
        conf.insert(0, "test_condition", condition)
        conf.insert(0, "language", lang)
        all_confusions.append(conf)

        # Augmentation rescue examples on the worst condition for clean-trained XLM-R.
        worst_clean = worst[(worst.language == lang) & (worst.model == "xlmr") & (worst.training_condition == "clean")].iloc[0]
        rescue_cond = worst_clean.test_condition
        a = clean_train[clean_train.test_condition == rescue_cond][[
            "source_row", "clean_utterance", "test_utterance", "true_intent", "true_intent_name",
            "majority_prediction", "majority_prediction_name", "agreement_count", "majority_correct"
        ]].rename(columns={
            "majority_prediction": "cleantrained_prediction",
            "majority_prediction_name": "cleantrained_prediction_name",
            "agreement_count": "cleantrained_agreement",
            "majority_correct": "cleantrained_correct",
        })
        b = mixed[mixed.test_condition == rescue_cond][[
            "source_row", "majority_prediction", "majority_prediction_name", "agreement_count", "majority_correct"
        ]].rename(columns={
            "majority_prediction": "mixed_prediction",
            "majority_prediction_name": "mixed_prediction_name",
            "agreement_count": "mixed_agreement",
            "majority_correct": "mixed_correct",
        })
        r = a.merge(b, on="source_row", validate="one_to_one")
        r = r[(~r.cleantrained_correct) & r.mixed_correct & r.cleantrained_prediction.notna() & r.mixed_prediction.notna()].copy()
        r["language"] = lang
        r["test_condition"] = rescue_cond
        r["priority"] = (r.cleantrained_agreement == 3).astype(int) + (r.mixed_agreement == 3).astype(int)
        r = r.sort_values(["priority", "mixed_agreement"], ascending=False).head(8)
        rescue_examples.append(r[[
            "language", "source_row", "test_condition", "clean_utterance", "test_utterance", "true_intent_name",
            "cleantrained_prediction_name", "mixed_prediction_name", "cleantrained_agreement", "mixed_agreement"
        ]])

    pd.concat(all_errors, ignore_index=True).to_csv(tables_dir / "08_xlmr_mixed_error_examples.csv", index=False)
    pd.concat(all_confusions, ignore_index=True).to_csv(tables_dir / "09_xlmr_mixed_top_confusions.csv", index=False)
    pd.concat(rescue_examples, ignore_index=True).to_csv(tables_dir / "10_xlmr_augmentation_rescue_examples.csv", index=False)


def verify_metrics(handoff: Path, all_metrics: pd.DataFrame, tables_dir: Path):
    records = []
    labels = list(range(60))
    metric_index = all_metrics.set_index(["language", "model", "training_condition", "seed", "test_condition"])
    for p in sorted((handoff / "outputs" / "predictions").glob("*.csv")):
        df = pd.read_csv(p)
        for condition, g in df.groupby("test_condition"):
            key = (g.language.iloc[0], g.model.iloc[0], g.training_condition.iloc[0], int(g.seed.iloc[0]), condition)
            expected = metric_index.loc[key]
            acc = accuracy_score(g.true_intent, g.predicted_intent)
            f1 = f1_score(g.true_intent, g.predicted_intent, labels=labels, average="macro", zero_division=0)
            records.append({
                "language": key[0], "model": key[1], "training_condition": key[2], "seed": key[3], "test_condition": key[4],
                "accuracy_recomputed": acc, "accuracy_saved": float(expected.accuracy),
                "macro_f1_recomputed": f1, "macro_f1_saved": float(expected.macro_f1),
                "accuracy_abs_diff": abs(acc - float(expected.accuracy)),
                "macro_f1_abs_diff": abs(f1 - float(expected.macro_f1)),
            })
    out = pd.DataFrame(records)
    out.to_csv(tables_dir / "11_metric_verification.csv", index=False)
    max_acc = out.accuracy_abs_diff.max()
    max_f1 = out.macro_f1_abs_diff.max()
    if max_acc > 1e-12 or max_f1 > 1e-12:
        raise AssertionError(f"Metric verification mismatch: acc {max_acc}, f1 {max_f1}")
    return out


def audit_corruptions(data_root: Path, tables_dir: Path):
    try:
        from rapidfuzz.distance import Levenshtein
    except Exception:
        return None
    rows = []
    corrupt_root = data_root / "corrupted_test"
    if not corrupt_root.exists():
        return None
    for lang in LANG_ORDER:
        for typo in TYPO_ORDER:
            for level in NOISE_ORDER:
                p = corrupt_root / lang / f"{typo}_{level:.2f}.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                clean = df.clean_utterance.fillna("").astype(str)
                noisy = df.utterance.fillna("").astype(str)
                distances = np.array([Levenshtein.distance(a, b) for a, b in zip(clean, noisy)], dtype=float)
                clean_lengths = clean.str.len().to_numpy()
                norm = distances / np.maximum(clean_lengths, 1)
                rows.append({
                    "language": lang,
                    "typo_type": typo,
                    "noise_level": level,
                    "rows": len(df),
                    "unchanged_rows": int((clean == noisy).sum()),
                    "empty_noisy_rows": int((noisy.str.len() == 0).sum()),
                    "mean_levenshtein_distance": float(distances.mean()),
                    "mean_normalized_edit_distance": float(norm.mean()),
                    "median_normalized_edit_distance": float(np.median(norm)),
                })
    audit = pd.DataFrame(rows)
    audit.to_csv(tables_dir / "12_corruption_audit.csv", index=False)
    return audit


def write_summary(out_dir: Path, tables: dict[str, pd.DataFrame], checks: dict, audit: pd.DataFrame | None):
    overall = tables["overall"]
    bylang = tables["by_language"]
    aug = tables["augmentation"]
    bytype = tables["by_type"]

    def overall_row(model, train):
        return overall[(overall.model == model) & (overall.training_condition == train)].iloc[0]

    word = overall_row("word_tfidf", "clean")
    char = overall_row("char_tfidf", "clean")
    xc = overall_row("xlmr", "clean")
    xm = overall_row("xlmr", "mixed")

    xlm_aug = aug[aug.model == "xlmr"]
    biggest = xlm_aug.loc[xlm_aug.noisy_f1_gain_mixed_minus_clean.idxmax()]
    english = xlm_aug[xlm_aug.language == "en-US"].iloc[0]
    type_avg = bytype[(bytype.model == "xlmr") & (bytype.training_condition == "mixed")].groupby("typo_type", as_index=False).drop_mean.mean()
    hardest_type = type_avg.loc[type_avg.drop_mean.idxmax()]

    lines = [
        "# Part 03 analysis notes",
        "",
        "These notes are generated from Sun's frozen training handoff. They are descriptive, not significance-test claims.",
        "",
        "## What was checked",
        f"- {checks['run_groups']} completed run groups, {checks['prediction_files']} prediction files, and {checks['test_metric_rows']} run-condition metric rows.",
        "- Every run has 13 test conditions: clean plus 12 corrupted conditions.",
        "- Report the mean and sample standard deviation across seeds 2028, 2029, and 2030. Do not select a best seed.",
        "",
        "## Main findings",
        f"- Explicit baseline, word TF-IDF + Logistic Regression trained on clean data: {word.clean_f1:.2f} clean macro-F1 and {word.noisy_f1:.2f} mean corrupted-test macro-F1 (equal-weight average over the four languages).",
        f"- Character TF-IDF trained on clean data is much more robust than the word baseline: {char.noisy_f1:.2f} mean corrupted-test macro-F1, with a {char.robustness_drop:.2f}-point clean-to-noisy drop versus {word.robustness_drop:.2f} points for the word baseline.",
        f"- Clean-trained XLM-R has the strongest clean score ({xc.clean_f1:.2f}) but a large average robustness drop ({xc.robustness_drop:.2f} points). Its noisy average ({xc.noisy_f1:.2f}) is only slightly above clean-trained character TF-IDF ({char.noisy_f1:.2f}).",
        f"- Mixed/augmented XLM-R is the strongest system on corrupted inputs: {xm.noisy_f1:.2f} mean corrupted-test macro-F1, with an {xm.robustness_drop:.2f}-point drop.",
        f"- The largest XLM-R noisy gain from augmentation is in {LANG_LABEL[biggest.language]}: +{biggest.noisy_f1_gain_mixed_minus_clean:.2f} macro-F1 points.",
        f"- Augmentation is not free: English XLM-R clean macro-F1 changes by {english.clean_f1_change_mixed_minus_clean:+.2f} points while its noisy macro-F1 changes by {english.noisy_f1_gain_mixed_minus_clean:+.2f} points.",
        f"- For mixed XLM-R, {hardest_type.typo_type} is the most damaging corruption type on average across languages/levels, with a mean drop of {hardest_type.drop_mean:.2f} points.",
        "- Increasing the nominal corruption level generally hurts performance, but Simplified Chinese is not strictly monotonic at 5% vs 10% in the current fixed synthetic corruptions.",
        "",
        "## Important limitations to state in the report",
        "- The current generator is controlled synthetic character corruption, not a validated model of natural human typos. It does not model keyboard adjacency, Vietnamese input-method/diacritic errors, Chinese IME/homophone errors, or German compound spelling.",
        "- The substitution character pools were built from all retained splits, including held-out text. Labels are not used, but the corruption process is therefore not strictly train-text-only.",
        "- Swap selection can use overlapping adjacent pairs. Do not describe swaps as guaranteed disjoint adjacent transpositions.",
        "- Some rows marked corrupted are unchanged, and a few deletion outputs are empty.",
        "- The same frozen noisy datasets are reused for all three training seeds. Standard deviations therefore describe training variability conditional on one corruption realization, not uncertainty over possible typo samples.",
        "- Mixed training doubles the number of training examples. Its gain cannot be attributed purely to augmentation independent of extra optimization exposure.",
        "- No hyperparameter search, compute-matched augmentation control, truncation audit, confidence intervals, or significance tests were run.",
        "",
        "## Strongest report story",
        "1. Start from the explicit word-level TF-IDF + Logistic Regression clean baseline.",
        "2. Show that character features are substantially more robust to character corruption than word features.",
        "3. Show that XLM-R is strongest on clean data but clean-trained XLM-R can be surprisingly brittle under synthetic corruption.",
        "4. Show that typo-augmented/mixed training strongly improves XLM-R robustness, while noting the small clean-performance trade-off in English.",
        "5. Finish with language/corruption differences and a careful limitations discussion, especially the realism issue raised by the course feedback.",
    ]
    if audit is not None and not audit.empty:
        unchanged = int(audit.unchanged_rows.sum())
        empty = int(audit.empty_noisy_rows.sum())
        lines += [
            "",
            "## Corruption audit from exported data",
            f"- Across all language/type/level files, {unchanged} generated rows are textually unchanged and {empty} noisy rows are empty.",
            "- See `tables/12_corruption_audit.csv` for actual edit-distance summaries by language, operation and nominal level.",
        ]
    (out_dir / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Part 03 analysis for MASSIVE typo robustness project")
    ap.add_argument("--handoff", type=Path, required=True, help="Path to Sun's handoff_03_20261016 directory")
    ap.add_argument("--data-root", type=Path, default=None, help="Optional path to EssentialsNLP-main/exported_data for corruption audit")
    ap.add_argument("--output", type=Path, default=Path("analysis_outputs"))
    ap.add_argument("--verify-metrics", action="store_true", help="Recompute all 936 metrics from the ~500MB prediction files")
    args = ap.parse_args()

    handoff = args.handoff.resolve()
    out_dir = ensure_dir(args.output.resolve())
    fig_dir = ensure_dir(out_dir / "figures")
    tables_dir = ensure_dir(out_dir / "tables")

    all_metrics, three_seed, training, intent_mapping = load_inputs(handoff)
    checks = integrity_checks(handoff, all_metrics, three_seed)
    (out_dir / "integrity_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    tables = make_tables(all_metrics, three_seed, out_dir)
    plot_overall(tables["overall"], fig_dir)
    plot_augmentation(tables["augmentation"], fig_dir)
    plot_noise_curves(tables["by_level"], fig_dir)
    plot_typo_type(tables["by_type"], fig_dir)
    plot_language_robustness(tables["by_language"], fig_dir)
    plot_baseline_comparison(tables["baseline"], fig_dir)

    error_analysis(handoff, tables_dir, tables["worst"])
    audit = audit_corruptions(args.data_root.resolve(), tables_dir) if args.data_root else None

    if args.verify_metrics:
        verify_metrics(handoff, all_metrics, tables_dir)

    write_summary(out_dir, tables, checks, audit)
    print(f"Analysis complete: {out_dir}")
    print("Open RESULTS_SUMMARY.md first, then tables/ and figures/.")


if __name__ == "__main__":
    main()
