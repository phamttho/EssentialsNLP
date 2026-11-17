from .common import *
import importlib.metadata as metadata
import sys
import shutil

def check(verify_weights=False):
    import torch
    from transformers import AutoTokenizer,Trainer,TrainingArguments,EarlyStoppingCallback
    assert Path(sys.prefix).resolve()==ROOT/"environment"
    arches=torch._C._cuda_getArchFlags()
    assert "sm_120" in arches,"PyTorch lacks RTX 5090 support"
    assert CONFIG["seeds"]==[2028,2029,2030]
    assert CONFIG["xlmr"]["max_epochs"]==30 and CONFIG["xlmr"]["patience"]==10
    assert len(all_runs())==72
    info=json.loads((ROOT/"cache/pretrained/metadata.json").read_text())
    snapshot=Path(info["snapshot"]).resolve()
    assert snapshot.is_relative_to(ROOT/"cache")
    assert (snapshot/"model.safetensors").is_file()
    if verify_weights:
        for name,digest in info["sha256"].items():
            assert sha(snapshot/name)==digest, f"Pretrained hash mismatch: {name}"
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    assert len(tokenizer(["hello","你好","xin chào","Guten Tag"])["input_ids"])==4
    for key in ["HF_HOME","HF_HUB_CACHE","HF_DATASETS_CACHE","TORCH_HOME","PIP_CACHE_DIR",
                "MPLCONFIGDIR","XDG_CACHE_HOME","TRITON_CACHE_DIR","CUDA_CACHE_PATH","TMPDIR"]:
        assert Path(os.environ[key]).resolve().is_relative_to(ROOT)
    report={"status":"PASS","python":sys.version,"prefix":sys.prefix,
        "versions":{x:metadata.version(x) for x in ["torch","transformers","accelerate","scikit-learn","pandas","jieba"]},
        "cuda_build":torch.version.cuda,"architectures":arches,"snapshot":str(snapshot),
        "weight_hashes_verified":verify_weights,"disk_free_gib":shutil.disk_usage(ROOT).free/1024**3}
    assert report["disk_free_gib"]>=CONFIG["minimum_free_disk_gib"]
    save_json(ROOT/"outputs/manifests/environment_check.json",report)
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__":
    check(verify_weights=True)
