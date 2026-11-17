from .common import *
import argparse
import fcntl
import subprocess
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor,as_completed

def gpu_free(gpu):
    assert str(gpu) in CONFIG["gpu_languages"]
    uuid=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    memory=int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
    if uuid in processes or memory>256:
        raise RuntimeError(f"GPU {gpu} is occupied; refusing to share it")

def check_disk():
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<CONFIG["minimum_free_disk_gib"]:
        raise RuntimeError(f"Only {free:.1f} GiB free; below configured disk threshold")

def run_one(args,mark,gpu=None):
    from .export import completed_valid
    from .train import cleanup_checkpoints
    rid=run_id(*args)
    if completed_valid(args,mark):
        cleanup_checkpoints(ROOT/"checkpoints"/rid,True)
        print("SKIP",rid,flush=True);return
    assert fingerprint()==mark,"Source/config/data changed while runner is active"
    check_disk()
    if gpu is not None:gpu_free(gpu)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"" if gpu is None else str(gpu),"PYTHONUNBUFFERED":"1"}
    cmd=[sys.executable,"-B","-m","src.train","--language",args[0],"--model",args[1],
         "--condition",args[2],"--seed",str(args[3]),"--fingerprint",mark]
    with open(ROOT/"logs"/f"{rid}.log","a",buffering=1) as log:
        log.write(f"\nSTART {time.ctime()}\n")
        print("START",rid,"GPU",gpu,flush=True)
        proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"Failed {rid}; read logs/{rid}.log")
    print("DONE",rid,flush=True)

def gpu_queue(gpu,mark):
    lang=CONFIG["gpu_languages"][str(gpu)]
    for seed in CONFIG["seeds"]:
        for condition in CONFIG["conditions"]:
            run_one((lang,"xlmr",condition,seed),mark,gpu)

def cpu_queue(worker,mark):
    jobs=[a for a in all_runs() if a[1]!="xlmr"]
    for args in jobs[worker::CONFIG["cpu_workers"]]:
        run_one(args,mark)

def run_all(lock_fd=None):
    from .check_data import validate
    from .check_environment import check
    from .export import aggregate
    lock=os.fdopen(lock_fd,"a") if lock_fd is not None else open(ROOT/"run.lock","a")
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={"pid":os.getpid(),"started":time.time(),"status":"running","expected_runs":72}
        save_json(ROOT/"outputs/manifests/runner.json",state)
        try:
            validate();check()
            mark=fingerprint()
            for gpu in CONFIG["gpu_languages"]:gpu_free(int(gpu))
            failures=[]
            with ThreadPoolExecutor(max_workers=4+CONFIG["cpu_workers"]) as pool:
                jobs={pool.submit(gpu_queue,int(gpu),mark):f"GPU {gpu}" for gpu in CONFIG["gpu_languages"]}
                jobs.update({pool.submit(cpu_queue,w,mark):f"CPU queue {w}" for w in range(CONFIG["cpu_workers"])})
                for future in as_completed(jobs):
                    try:future.result()
                    except Exception as exc:
                        failures.append(f"{jobs[future]}: {exc}")
                        print("QUEUE FAILED",failures[-1],flush=True)
            if failures:raise RuntimeError("; ".join(failures))
            aggregate()
            state.update(status="completed",finished=time.time())
        except BaseException as exc:
            state.update(status="failed",finished=time.time(),error=str(exc))
            raise
        finally:
            save_json(ROOT/"outputs/manifests/runner.json",state)

def active_workers():
    found=[]
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and (process_alive(int(path.name),"src.train") or process_alive(int(path.name),"src.runner")):
            if int(path.name)!=os.getpid():found.append(int(path.name))
    return found

def start():
    for name in ["logs","outputs/manifests","locks"]:
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    # Pass the held lock to the child so there is no gap for duplicate launchers.
    with open(ROOT/"run.lock","a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("A project runner is already active")
        busy=active_workers()
        if busy:raise RuntimeError(f"Existing project worker processes: {busy}; inspect status first")
        for gpu in CONFIG["gpu_languages"]:gpu_free(int(gpu))
        check_disk()
        with open(ROOT/"logs/runner.log","a") as log:
            proc=subprocess.Popen([sys.executable,"-B","-m","src.runner","run","--lock-fd",str(lock.fileno())],
                cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONUNBUFFERED":"1"},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True,pass_fds=(lock.fileno(),))
        time.sleep(3)
        if proc.poll() is not None:
            raise RuntimeError("Runner exited during startup; read logs/runner.log")
        print(f"Started detached runner PID {proc.pid}. Check scripts/status.py for progress.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["start","run"])
    p.add_argument("--lock-fd",type=int)
    a=p.parse_args()
    if a.command=="start":start()
    else:run_all(a.lock_fd)
