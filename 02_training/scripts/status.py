from pathlib import Path
import json
import subprocess
import os
root=Path(__file__).resolve().parents[1]
folder=root/"outputs/manifests"
def alive(pid,module):
    try:
        p=Path("/proc")/str(pid)
        return module.encode() in (p/"cmdline").read_bytes().split(b"\x00") and (p/"cwd").resolve()==root
    except OSError:return False
def tail(path,n=5):
    if not path.exists():return ""
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-16384))
        return "\n".join(stream.read().decode("utf-8",errors="replace").splitlines()[-n:])
state_path=folder/"runner.json"
if state_path.exists():
    state=json.loads(state_path.read_text())
    print("Runner:",state["status"],"PID:",state["pid"],"alive:",alive(state["pid"],"src.runner"))
    if "error" in state:print("ERROR:",state["error"])
else:print("Formal training has not been started.")
counts={"completed":0,"running":0,"failed":0,"interrupted":0}
for path in sorted(folder.glob("*__seed*.json")):
    state=json.loads(path.read_text());status=state["status"]
    if status=="running" and not alive(state["pid"],"src.train"):status="interrupted"
    counts[status]+=1
    if status!="completed":
        print(status,state["run_id"])
        print(tail(root/"logs"/(state["run_id"]+".log"),3))
counts["pending"]=72-sum(counts.values())
print("Progress:",counts,"/ 72 runs")
print(subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,utilization.gpu","--format=csv"],text=True))
print(tail(root/"logs/runner.log",8))
handoff=folder/"handoff.json"
if handoff.exists():print("Handoff:",json.loads(handoff.read_text())["status"])
