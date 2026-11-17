import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "01_data" / "exported_data"
for key, rel in {
    "HF_HOME":"cache/huggingface", "HF_HUB_CACHE":"cache/huggingface/hub",
    "HF_DATASETS_CACHE":"cache/huggingface/datasets", "TORCH_HOME":"cache/torch",
    "PIP_CACHE_DIR":"cache/pip", "MPLCONFIGDIR":"cache/matplotlib",
    "XDG_CACHE_HOME":"cache/xdg", "TRITON_CACHE_DIR":"cache/triton",
    "CUDA_CACHE_PATH":"cache/cuda", "TMPDIR":"tmp", "TMP":"tmp", "TEMP":"tmp",
}.items():
    path = ROOT / rel
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "4"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import json
import hashlib
import tempfile
tempfile.tempdir = str(ROOT / "tmp")
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

CONFIG = json.loads((ROOT / "configs/experiment.json").read_text())
def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)

def sha(path):
    digest = hashlib.sha256()
    with open(path,"rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def manifest():
    return json.loads((DATA / "metadata/data_manifest.json").read_text())

def label_mapping():
    return {int(k):v for k,v in json.loads((DATA / "metadata/intent_mapping.json").read_text()).items()}

def read_entry(entry):
    path = (DATA / entry["path"]).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError("Data path outside project")
    df = pd.read_csv(path, keep_default_na=False)
    assert len(df) == entry["rows"], f"Row count mismatch: {path}"
    assert {"utterance","intent"} <= set(df.columns), str(path)
    ids = label_mapping()
    if not pd.api.types.is_integer_dtype(df.intent):
        inverse = {v:k for k,v in ids.items()}
        if set(df.intent.astype(str)) <= set(inverse):
            df["intent"] = df.intent.map(inverse)
        else:
            df["intent"] = pd.to_numeric(df.intent, errors="raise")
    assert np.all(df.intent == df.intent.astype(int)), str(path)
    df["intent"] = df.intent.astype(int)
    assert set(df.intent) <= set(ids), f"Unknown label: {path}"
    assert df.utterance.map(lambda x:isinstance(x,str)).all()
    return df

def clean(lang, split):
    return read_entry(manifest()["clean"][lang][split])

def training(lang, condition):
    if condition == "clean":
        return clean(lang,"train")
    if condition != "mixed":
        raise ValueError(condition)
    return read_entry(manifest()["augmented"][lang])

def test_sets(lang):
    base = clean(lang, "test")
    base["source_row"] = np.arange(len(base))
    base["clean_utterance"] = base.utterance
    yield "clean", "none", 0.0, base
    for typo, levels in manifest()["corrupted_test"][lang].items():
        for level, entry in levels.items():
            yield f"{typo}_{level}", typo, float(level), read_entry(entry)

def metrics(y, pred):
    return {"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,labels=list(label_mapping()),average="macro",zero_division=0))}

def chinese_tokens(text):
    import jieba
    jieba.dt.tmp_dir = str(ROOT / "cache")
    return [x for x in jieba.lcut(text) if x.strip()]

def run_id(lang, model, condition, seed):
    assert seed in CONFIG["seeds"]
    return f"{lang}__{model}__{condition}__seed{seed}"

def model_dir(lang,model,condition,seed):
    return ROOT/"models"/lang/model/condition/f"seed{seed}"

def all_runs():
    return [(l,m,c,s) for l in CONFIG["languages"] for m in CONFIG["models"]
            for c in CONFIG["conditions"] for s in CONFIG["seeds"]]

def fingerprint():
    report=ROOT/"outputs/manifests/data_check.json"
    source={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/"src").glob("*.py"))}
    blob={"config":CONFIG,"data":json.loads(report.read_text())["data_sha256"],
          "source":source,"dependencies":sha(ROOT/"requirements.lock.txt"),
          "pretrained":json.loads((ROOT/"cache/pretrained/metadata.json").read_text())}
    return hashlib.sha256(json.dumps(blob,sort_keys=True).encode()).hexdigest()

def model_hashes(directory):
    files=[p for p in directory.iterdir() if p.is_file() and p.name not in ("training_complete.json","training_history.json")]
    return {p.name:sha(p) for p in sorted(files)}

def process_alive(pid, module):
    try:
        proc=Path("/proc")/str(pid)
        return module.encode() in (proc/"cmdline").read_bytes().split(b"\x00") and (proc/"cwd").resolve()==ROOT
    except (OSError,ProcessLookupError):
        return False
