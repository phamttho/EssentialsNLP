"""Two-phase validation-only search, frozen selection, then test export."""
from base import *
import argparse,subprocess,fcntl,threading,queue,traceback,statistics
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
PY=OLD/"environment/bin/python"
PLAN=json.loads((ROOT/"plan.json").read_text())

def gpu_free(gpu):
    uuid=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    procs=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    mem=int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True))
    if uuid in procs or mem>256:raise RuntimeError(f"GPU {gpu} occupied; refusing to share")
def old_audit():
    # Read and hash, never import old code or mutate any original artifacts.
    for rel,digest in input_hashes().items():
        path=(DATA/rel).resolve();assert path.is_relative_to(DATA.resolve())
        assert sha(path)==digest,f"Frozen input changed: {rel}"
    blob={"config":CONFIG,"data":input_hashes(),
      "source":{p.relative_to(OLD).as_posix():sha(p) for p in sorted((OLD/"src").glob("*.py"))},
      "dependencies":sha(OLD/"requirements.lock.txt"),
      "pretrained":json.loads((OLD/"cache/pretrained/metadata.json").read_text())}
    oldfp=hashlib.sha256(json.dumps(blob,sort_keys=True).encode()).hexdigest()
    pm=blob["pretrained"]
    for name,digest in pm["sha256"].items():assert sha(snapshot()/name)==digest,f"Pretrained mismatch {name}"
    for l in LANGUAGES:
        for m in MODELS:
            for c in CONDITIONS:
                for s in SEEDS:
                    r=rid(l,m,c,s);mft=json.loads((OLD/"outputs/manifests"/(r+".json")).read_text())
                    assert mft["status"]=="completed" and mft["fingerprint"]==oldfp,r
                    val=json.loads((OLD/"outputs/validation_metrics"/(r+".json")).read_text())
                    assert all(0<=val[k]<=1 for k in ("accuracy","macro_f1"))
                    if m=="xlmr":assert abs(val["macro_f1"]-mft["training_summary"]["best_validation_macro_f1"])<1e-9
    return oldfp
def getlock(path):
    h=open(path,"a");fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def record_path(stage,job):
    l,m,c,s,v=job
    return ROOT/"records"/stage/(cid(l,m,c,s,v)+".json")
def valid_record(stage,job):
    path=record_path(stage,job)
    if not path.exists():return False
    rec=json.loads(path.read_text())
    assert rec.get("fingerprint")==experiment_fingerprint(),"Experiment changed; do not mix runs"
    if rec.get("status")!="completed":return False
    for rel,digest in rec["files"].items():
        p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT)
        assert p.exists() and sha(p)==digest,f"Output changed: {rel}"
    return True
def assert_locked():
    lock=ROOT/"state/experiment.json"
    assert lock.exists(),"Experiment must be locked by runner first"
    assert json.loads(lock.read_text())["fingerprint"]==experiment_fingerprint(),"Source/config/evidence changed during experiment"
def assert_selected(job):
    p=ROOT/"results/selection.json"
    assert p.exists(),"Final test export requires a frozen selection"
    selection=json.loads(p.read_text())
    assert selection["fingerprint"]==experiment_fingerprint() and selection["locked_before_test"]
    l,m,c,s,v=job
    match=[x for x in selection["selected"] if x["language"]==l and x["model"]==m]
    assert len(match)==1 and match[0]["value"]==v,"Job is not the selected parameter"
def record(stage,job,summary,files,**extra):
    assert_locked()
    save_json(record_path(stage,job),{"status":"completed","fingerprint":experiment_fingerprint(),"job":job,
      "summary":summary,"files":{p.relative_to(ROOT).as_posix():sha(p) for p in files},"finished":time.time(),**extra})
def validation_export(job,predict,summary,dest):
    l,m,c,s,v=job;df=clean(l,"validation");pred=np.asarray(predict(df.utterance.tolist()),dtype=int)
    assert len(pred)==len(df) and set(pred)<=set(range(60))
    dest.mkdir(parents=True,exist_ok=True)
    out=dest/"validation_predictions.csv.gz"
    pd.DataFrame({"source_row":np.arange(len(df)),"true_intent":df.intent,"predicted_intent":pred}).to_csv(out,index=False,compression="gzip")
    back=pd.read_csv(out,keep_default_na=False)
    assert len(back)==len(df) and np.array_equal(back.source_row,np.arange(len(df)))
    assert np.array_equal(back.true_intent,df.intent) and np.array_equal(back.predicted_intent,pred)
    scores=metrics(back.true_intent,back.predicted_intent)
    for k in scores:assert abs(scores[k]-summary[k])<1e-9,(k,scores,summary)
    save_json(dest/"validation_metrics.json",scores)
    return [out,dest/"validation_metrics.json"]
def process_job(stage,job):
    from fit import fit_job
    from evaluate import export_and_verify,import_legacy
    assert_locked()
    l,m,c,s,v=job
    assert l in LANGUAGES and m in MODELS and c in CONDITIONS and s in SEEDS and v in GRID[m]
    if stage=="final":assert_selected(job)
    j=cid(*job);work=ROOT/"work"/(stage+"__"+j)
    lk=getlock(ROOT/"locks"/(stage+"__"+j+".lock"))
    try:
        if valid_record(stage,job):clear_work(work);return
        is_old=v==ORIGINAL[m];started=time.time()
        dest=ROOT/("search" if stage=="search" else "results/final")/j
        # Retries restart incomplete fits; stale weights only within this specific job are discarded.
        clear_work(work)
        if is_old:
            assert stage=="final"
            info=import_legacy(l,m,c,s,v,dest)
            val=json.loads((OLD/"outputs/validation_metrics"/(rid(l,m,c,s)+".json")).read_text())
            record(stage,job,val,list(dest.glob("*")),provenance="original_run_reused",export=info,started=started)
            return
        work.mkdir(parents=True);predict,summary=fit_job(l,m,c,s,v,work)
        vf=validation_export(job,predict,summary,dest)
        hist=work/"training_history.json"
        if hist.exists():
            shutil.copy2(hist,dest/"training_history.json");vf.append(dest/"training_history.json")
        if (work/"validation_history.json").exists():
            shutil.copy2(work/"validation_history.json",dest/"validation_history.json");vf.append(dest/"validation_history.json")
        if stage=="search":
            record(stage,job,summary,vf,provenance="new_validation_only",started=started)
        else:
            info=export_and_verify(l,m,c,s,v,predict,dest)
            search=json.loads(record_path("search",job).read_text())
            record(stage,job,summary,list(dest.glob("*")),provenance="selected_parameter_retrained",
              search_macro_f1=search["summary"]["macro_f1"],
              refit_validation_difference=summary["macro_f1"]-search["summary"]["macro_f1"],export=info,started=started)
        assert valid_record(stage,job)
        clear_work(work)
        print("COMPLETED",stage,j,flush=True)
    finally:lk.close()
def import_original():
    for l in LANGUAGES:
        for m in MODELS:
            for c in CONDITIONS:
                for s in SEEDS:
                    job=(l,m,c,s,ORIGINAL[m])
                    if valid_record("search",job):continue
                    r=rid(l,m,c,s);vp=OLD/"outputs/validation_metrics"/(r+".json")
                    val=json.loads(vp.read_text())
                    dest=ROOT/"search"/cid(*job);dest.mkdir(parents=True,exist_ok=True)
                    save_json(dest/"validation_metrics.json",val)
                    mft=OLD/"outputs/manifests"/(r+".json")
                    record("search",job,val,[dest/"validation_metrics.json"],provenance="original_validation_reused",
                      original_validation_sha256=sha(vp),original_manifest_sha256=sha(mft))
def all_search_jobs():
    return [(l,m,c,s,v) for l in LANGUAGES for m in MODELS for v in GRID[m] if v!=ORIGINAL[m]
      for c in CONDITIONS for s in SEEDS]
def select_parameters():
    assert_locked()
    rows=[];choices=[]
    for l in LANGUAGES:
        for m in MODELS:
            scores={}
            for v in GRID[m]:
                ss=[]
                for c in CONDITIONS:
                    for s in SEEDS:
                        job=(l,m,c,s,v);assert valid_record("search",job)
                        rec=json.loads(record_path("search",job).read_text())
                        score=float(rec["summary"]["macro_f1"]);ss.append(score)
                        rows.append(dict(language=l,model=m,condition=c,seed=s,value=v,validation_macro_f1=score,provenance=rec["provenance"]))
                assert len(ss)==6;scores[v]=statistics.mean(ss)
            best=sorted(scores,key=lambda v:(-scores[v],v!=ORIGINAL[m],GRID[m].index(v)))[0]
            choices.append(dict(language=l,model=m,value=best,parameter="learning_rate" if m=="xlmr" else "C",
              score=scores[best],candidate_means={str(v):s for v,s in scores.items()}))
    assert len(rows)==216 and len(choices)==12
    pd.DataFrame(rows).to_csv(ROOT/"results/search_validation_scores.csv",index=False)
    frozen={"fingerprint":experiment_fingerprint(),"selected":choices,"criterion":PLAN["selection"],"locked_before_test":True}
    path=ROOT/"results/selection.json"
    if path.exists():assert json.loads(path.read_text())==frozen,"Selection differs; refusing to overwrite"
    else:save_json(path,frozen)
    return choices

disk_lock=threading.Lock()
failed=threading.Event()
def reserve(key,gib):
    with disk_lock:
        p=ROOT/"state/reservations.json";ls=json.loads(p.read_text()) if p.exists() else {}
        free=shutil.disk_usage(ROOT).free/1024**3
        required=25+sum(x["gib"] for x in ls.values())+gib
        if free<required:raise RuntimeError(f"Disk safety pause: free {free:.2f}GiB, need {required:.2f}GiB. Re-run after freeing space; old data untouched.")
        ls[key]={"gib":gib};save_json(p,ls)
def release(key):
    with disk_lock:
        p=ROOT/"state/reservations.json";ls=json.loads(p.read_text());ls.pop(key,None);save_json(p,ls)
def run_queue(stage,jobs):
    queues={"gpu":queue.Queue(),"cpu":queue.Queue()}
    for j in jobs:
        if valid_record(stage,j):
            clear_work(ROOT/"work"/(stage+"__"+cid(*j)))
            save_json(ROOT/"state"/(stage+"__"+cid(*j)+".json"),{"status":"completed","job":j,"reused_on_restart":True})
            continue
        queues["gpu" if j[1]=="xlmr" and j[4]!=ORIGINAL[j[1]] else "cpu"].put(j)
    def worker(gpu):
        q=queues["gpu" if gpu is not None else "cpu"]
        while not failed.is_set():
            try:job=q.get_nowait()
            except queue.Empty:return
            key=stage+"__"+cid(*job)
            try:
                assert_locked()
                reserve(key,PLAN["gpu_reservation_gib"] if gpu is not None else PLAN["cpu_reservation_gib"])
                if gpu is not None:gpu_free(gpu)
                env={**os.environ,"CUDA_VISIBLE_DEVICES":"" if gpu is None else str(gpu),"PYTHONUNBUFFERED":"1"}
                args=[str(PY),"-B",str(ROOT/"run.py"),"job","--stage",stage,"--job",json.dumps(job)]
                save_json(ROOT/"state"/(key+".json"),{"status":"running","job":job,"gpu":gpu,"started":time.time()})
                with open(ROOT/"logs"/(key+".log"),"a",buffering=1) as log:
                    log.write("\nATTEMPT "+time.ctime()+"\n");print("START",key,"GPU",gpu,flush=True)
                    # Inherit the singleton lock so an orphaned worker prevents a second runner.
                    proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(RUN_LOCK.fileno(),))
                    save_json(ROOT/"state"/(key+".json"),{"status":"running","job":job,"gpu":gpu,"pid":proc.pid,"started":time.time()})
                    rc=proc.wait()
                if rc:raise RuntimeError(f"Job failed {key}; see logs/{key}.log")
                save_json(ROOT/"state"/(key+".json"),{"status":"completed","job":job,"gpu":gpu,"finished":time.time()})
            except BaseException as e:
                failed.set();save_json(ROOT/"state"/(key+".json"),{"status":"failed","job":job,"error":str(e)})
                print("ERROR",str(e),flush=True);raise
            finally:release(key);q.task_done()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(worker,g) for g in PLAN["gpus"]]+[pool.submit(worker,None) for _ in range(PLAN["cpu_workers"])]
        errors=[]
        for f in futures:
            try:f.result()
            except BaseException as e:errors.append(str(e))
    if errors:raise RuntimeError("; ".join(errors))
def preflight():
    oldfp=old_audit()
    for g in PLAN["gpus"]:gpu_free(g)
    guard()
    expected=25+len(PLAN["gpus"])*PLAN["gpu_reservation_gib"]+PLAN["cpu_workers"]*PLAN["cpu_reservation_gib"]
    free=shutil.disk_usage(ROOT).free/1024**3
    assert free>=expected,f"Need {expected}GiB to start full concurrency; found {free:.2f}"
    report={"status":"PASS","old_fingerprint":oldfp,"new_fingerprint":experiment_fingerprint(),
      "free_gib":free,"full_concurrency_required_gib":expected,"gpus":PLAN["gpus"],"time":time.time()}
    save_json(ROOT/"state/preflight.json",report);return report
def launch():
    lk=getlock(ROOT/"locks/runner.lock")
    logfile=open(ROOT/"logs/runner.log","a",buffering=1)
    env={**os.environ,"PYTHONUNBUFFERED":"1","CUDA_VISIBLE_DEVICES":""}
    proc=subprocess.Popen([str(PY),"-B",str(ROOT/"run.py"),"run","--lock-fd",str(lk.fileno())],
     cwd=ROOT,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lk.fileno(),),stdin=subprocess.DEVNULL)
    print(f"Started detached runner PID {proc.pid}. Use bash scripts/status.sh to confirm progress.")
    logfile.close();lk.close()
def run(lock_fd):
    global RUN_LOCK
    RUN_LOCK=os.fdopen(lock_fd,"a") if lock_fd is not None else getlock(ROOT/"locks/runner.lock")
    fcntl.flock(RUN_LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={"status":"running","pid":os.getpid(),"started":time.time(),"phase":"preflight"}
    save_json(ROOT/"state/runner.json",state)
    try:
        save_json(ROOT/"state/reservations.json",{})
        preflight()
        marker=ROOT/"state/experiment.json";fp=experiment_fingerprint()
        if marker.exists():assert json.loads(marker.read_text())["fingerprint"]==fp,"Locked experiment code/config changed"
        else:save_json(marker,{"fingerprint":fp,"created":time.time()})
        import_original()
        state["phase"]="validation_search";save_json(ROOT/"state/runner.json",state)
        run_queue("search",all_search_jobs())
        choices=select_parameters()
        state["phase"]="selected_refit_and_test";save_json(ROOT/"state/runner.json",state)
        final=[(x["language"],x["model"],c,s,x["value"]) for x in choices for c in CONDITIONS for s in SEEDS]
        run_queue("final",final)
        for j in final:assert valid_record("final",j)
        assert_locked()
        from evaluate import aggregate
        entries=[dict(language=j[0],model=j[1],condition=j[2],seed=j[3],value=j[4],dest=ROOT/"results/final"/cid(*j)) for j in final]
        aggregate(entries)
        state.update(status="completed",phase="complete",finished=time.time(),final_runs=72)
    except BaseException as e:
        state.update(status="failed",error=str(e),finished=time.time());traceback.print_exc();raise
    finally:save_json(ROOT/"state/runner.json",state);RUN_LOCK.close()
def status():
    p=ROOT/"state/runner.json"
    if not p.exists():print("Not launched.")
    else:
        s=json.loads(p.read_text());pid=s.get("pid")
        alive=pid is not None and Path(f"/proc/{pid}").exists()
        print(json.dumps({**s,"pid_exists":alive},indent=2))
    for st in ("search","final"):
        files=list((ROOT/"records"/st).glob("*.json"))
        print(st,"completed records",sum(json.loads(p.read_text()).get("status")=="completed" for p in files),
          "/ "+("216 (72 reused + 144 new)" if st=="search" else "72 selected runs"))
    for p in (ROOT/"state").glob("*__*.json"):
        x=json.loads(p.read_text())
        if x["status"] in ("running","failed"):
            print(x)
            if x["status"]=="running":
                tasklog=ROOT/"logs"/(p.stem+".log")
                if tasklog.exists():print("\n".join(tasklog.read_text(errors="replace").splitlines()[-2:]))
    print("Free GiB",round(shutil.disk_usage(ROOT).free/1024**3,2))
    subprocess.run(["nvidia-smi","--query-gpu=index,name,memory.used,utilization.gpu","--format=csv"])
    log=ROOT/"logs/runner.log"
    if log.exists():print("\nRecent log:\n"+"\n".join(log.read_text(errors="replace").splitlines()[-15:]))
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("action",choices=["start","run","job","status","preflight"])
    a.add_argument("--stage",choices=["search","final"]);a.add_argument("--job");a.add_argument("--lock-fd",type=int)
    args=a.parse_args()
    if args.action=="start":launch()
    elif args.action=="run":run(args.lock_fd)
    elif args.action=="job":process_job(args.stage,tuple(json.loads(args.job)))
    elif args.action=="status":status()
    else:print(json.dumps(preflight(),indent=2))
