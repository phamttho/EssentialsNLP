"""Small hardware check only, no production model or data training."""
from base import *
import subprocess,platform
from run import gpu_free
def main():
    for i in range(3):gpu_free(i)
    import torch
    assert torch.cuda.device_count()==3
    devices=[]
    for i in range(3):
        with torch.cuda.device(i):
            assert torch.cuda.is_bf16_supported()
            a=torch.randn((128,128),device=f"cuda:{i}",dtype=torch.bfloat16)
            b=a@a
            torch.cuda.synchronize(i);assert torch.isfinite(b).all()
            devices.append({"index":i,"name":torch.cuda.get_device_name(i),"bf16_check":"PASS"})
            del a,b;torch.cuda.empty_cache()
    import transformers,sklearn,pandas,accelerate
    result={"status":"PASS","python":platform.python_version(),"torch":torch.__version__,
      "transformers":transformers.__version__,"sklearn":sklearn.__version__,"pandas":pandas.__version__,
      "accelerate":accelerate.__version__,"gpus":devices,"production_training_started":False}
    save_json(ROOT/"state/hardware_check.json",result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
