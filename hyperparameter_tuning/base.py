"""Shared helpers. Never import the original src modules: they change cache paths."""
import os,sys,json,hashlib,tempfile,shutil,time
from pathlib import Path
from functools import lru_cache
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
EXPECTED=Path("/root/autodl-tmp/essentials_in_text_and_speech/02_training/hyperparameter_tuning")
if ROOT!=EXPECTED or ROOT.is_symlink(): raise RuntimeError("Unexpected experiment root")
OLD=ROOT.parent
DATA=OLD.parent/"01_data"/"exported_data"
for key,sub in {"HF_HOME":"cache/hf","HF_HUB_CACHE":"cache/hf/hub","HF_DATASETS_CACHE":"cache/datasets",
"TORCH_HOME":"cache/torch","PIP_CACHE_DIR":"cache/pip","XDG_CACHE_HOME":"cache/xdg",
"TRITON_CACHE_DIR":"cache/triton","CUDA_CACHE_PATH":"cache/cuda","MPLCONFIGDIR":"cache/matplotlib",
"TMPDIR":"tmp","TMP":"tmp","TEMP":"tmp"}.items():
    dst=ROOT/sub;dst.mkdir(parents=True,exist_ok=True);os.environ[key]=str(dst)
os.environ.update(PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
 TOKENIZERS_PARALLELISM="false",HF_HUB_DISABLE_TELEMETRY="1",HF_HUB_DISABLE_XET="1")
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="4"
tempfile.tempdir=str(ROOT/"tmp")
for d in ("logs","state","records","results","work","locks"): (ROOT/d).mkdir(exist_ok=True)
LANGUAGES=["en-US","vi-VN","de-DE","zh-CN"]
MODELS=["word_tfidf","char_tfidf","xlmr"]
CONDITIONS=["clean","mixed"];SEEDS=[2028,2029,2030]
GRID={"word_tfidf":[0.1,1.,10.],"char_tfidf":[0.1,1.,10.],"xlmr":[1e-5,2e-5,3e-5]}
ORIGINAL={"word_tfidf":1.,"char_tfidf":1.,"xlmr":2e-5}
CONFIG=json.loads((OLD/"configs/experiment.json").read_text())
def save_json(path,obj):
    path=Path(path);assert path.resolve().is_relative_to(ROOT),"Write outside tuning root"
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def snapshot():
    p=Path(json.loads((OLD/"cache/pretrained/metadata.json").read_text())["snapshot"]).resolve()
    assert p.is_relative_to((OLD/"cache/pretrained").resolve());return p
def manifest():return json.loads((DATA/"metadata/data_manifest.json").read_text())
def label_mapping():return {int(k):v for k,v in json.loads((DATA/"metadata/intent_mapping.json").read_text()).items()}
def read_entry(entry):
    import pandas as pd
    import numpy as np
    p=(DATA/entry["path"]).resolve();assert p.is_relative_to(DATA.resolve())
    expected=input_hashes().get(p.relative_to(DATA.resolve()).as_posix())
    assert expected is not None and sha(p)==expected,f"Frozen data changed: {p.name}"
    df=pd.read_csv(p,keep_default_na=False);assert len(df)==entry["rows"]
    ids=label_mapping()
    if not pd.api.types.is_integer_dtype(df.intent):
        inv={v:k for k,v in ids.items()}
        df["intent"]=df.intent.map(inv) if set(df.intent.astype(str))<=set(inv) else pd.to_numeric(df.intent,errors="raise")
    assert np.all(df.intent==df.intent.astype(int));df["intent"]=df.intent.astype(int)
    assert set(df.intent)<=set(ids)
    assert df.utterance.map(lambda x:isinstance(x,str)).all()
    return df
def clean(lang,split):return read_entry(manifest()["clean"][lang][split])
def training(lang,cond):
    assert cond in CONDITIONS
    return clean(lang,"train") if cond=="clean" else read_entry(manifest()["augmented"][lang])
def test_sets(lang):
    import numpy as np
    b=clean(lang,"test");b["source_row"]=np.arange(len(b));b["clean_utterance"]=b.utterance
    yield "clean","none",0.0,b
    for typo,levels in manifest()["corrupted_test"][lang].items():
        for level,e in levels.items():yield f"{typo}_{level}",typo,float(level),read_entry(e)
def metrics(y,pred):
    from sklearn.metrics import accuracy_score,f1_score
    return {"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,labels=list(range(60)),average="macro",zero_division=0))}
def chinese_tokens(t):
    import jieba
    jieba.dt.tmp_dir=str(ROOT/"cache")
    return [x for x in jieba.lcut(t) if x.strip()]
def rid(lang,model,cond,seed):return f"{lang}__{model}__{cond}__seed{seed}"
def cid(lang,model,cond,seed,value):return rid(lang,model,cond,seed)+"__value"+format(value,".8g")
def guard():
    leases=ROOT/"state/reservations.json"
    reservations=json.loads(leases.read_text()) if leases.exists() else {}
    required=(25+sum(float(x["gib"]) for x in reservations.values()))*1024**3
    free=shutil.disk_usage(ROOT).free
    if free<required:raise RuntimeError(f"Disk safety pause: {free/1024**3:.2f} GiB free; need {required/1024**3:.2f}; no old files will be deleted")
def clear_work(p):
    """Only a specific child of new work/ can be removed, never old weights."""
    p=Path(p);parent=ROOT/"work"
    assert p.parent==parent and p!=parent and not p.is_symlink()
    assert p.resolve().parent==parent.resolve() and p.name not in ("",".","..")
    if p.exists():shutil.rmtree(p)
def input_hashes():
    p=OLD/"outputs/manifests/data_check.json"
    return json.loads(p.read_text())["data_sha256"]
def experiment_fingerprint():
    src={p.name:sha(p) for p in sorted(ROOT.glob("*.py"))}
    payload={"code":src,"plan":json.loads((ROOT/"plan.json").read_text()),
     "old_config":sha(OLD/"configs/experiment.json"),"old_data_report":sha(OLD/"outputs/manifests/data_check.json"),
     "model_metadata":sha(OLD/"cache/pretrained/metadata.json"),"dependencies":sha(OLD/"requirements.lock.txt"),
     "old_validation":{p.name:sha(p) for p in sorted((OLD/"outputs/validation_metrics").glob("*.json"))}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
