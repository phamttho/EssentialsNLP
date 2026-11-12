from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "figure_source_three_seed_summary.csv"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Figure 1: final selected-parameter clean vs mean noisy macro-F1.
clean = (df[df["test_condition"] == "clean"]
         .groupby(["model", "training_condition"], as_index=False)["macro_f1_mean"].mean())
noisy = (df[df["test_condition"] != "clean"]
         .groupby(["language", "model", "training_condition"], as_index=False)["macro_f1_mean"].mean()
         .groupby(["model", "training_condition"], as_index=False)["macro_f1_mean"].mean())
summary = clean.merge(noisy, on=["model", "training_condition"], suffixes=("_clean", "_noisy"))
order = [("word_tfidf","clean"),("word_tfidf","mixed"),("char_tfidf","clean"),("char_tfidf","mixed"),("xlmr","clean"),("xlmr","mixed")]
labels = ["Word TF-IDF + LR\nclean","Word TF-IDF + LR\nmixed","Char TF-IDF + LR\nclean","Char TF-IDF + LR\nmixed","XLM-R-base\nclean","XLM-R-base\nmixed"]
clean_vals, noisy_vals = [], []
for model, condition in order:
    row = summary[(summary.model == model) & (summary.training_condition == condition)].iloc[0]
    clean_vals.append(row.macro_f1_mean_clean * 100)
    noisy_vals.append(row.macro_f1_mean_noisy * 100)

x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.bar(x-w/2, clean_vals, w, label="Clean test")
ax.bar(x+w/2, noisy_vals, w, label="Mean over 12 corrupted tests")
ax.set_ylabel("Macro-F1 (%)")
ax.set_title("Selected-parameter performance: clean vs corrupted test")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right")
ax.set_ylim(0, 100); ax.legend(); fig.tight_layout()
fig.savefig(OUT / "01_selected_clean_vs_noisy_macro_f1.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Figure 2: final selected mixed XLM-R performance by nominal noise level.
sub = df[(df.model == "xlmr") & (df.training_condition == "mixed") & (df.test_condition != "clean")]
curve = sub.groupby(["language", "noise_level"], as_index=False)["macro_f1_mean"].mean()
names = {"en-US":"English", "vi-VN":"Vietnamese", "de-DE":"German", "zh-CN":"Simplified Chinese"}
fig, ax = plt.subplots(figsize=(8.5, 5.3))
for lang in ["en-US", "vi-VN", "de-DE", "zh-CN"]:
    d = curve[curve.language == lang].sort_values("noise_level")
    ax.plot(d.noise_level * 100, d.macro_f1_mean * 100, marker="o", label=names[lang])
ax.set_xlabel("Nominal corruption level (%)")
ax.set_ylabel("Macro-F1 (%)")
ax.set_title("Selected mixed XLM-R: performance as noise increases")
ax.set_xticks([5,10,20]); ax.set_ylim(0,100); ax.legend(); fig.tight_layout()
fig.savefig(OUT / "02_selected_xlmr_noise_curve.png", dpi=220, bbox_inches="tight")
plt.close(fig)
