import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from src import common, train, export, runner
from transformers import EarlyStoppingCallback, TrainerState, TrainerControl

class Contracts(unittest.TestCase):
    def test_matrix(self):
        jobs=list(common.all_runs())
        self.assertEqual(len(set(jobs)),72)
        self.assertEqual(sum(j[1]=="xlmr" for j in jobs),24)
        self.assertEqual(sum(j[1]!="xlmr" for j in jobs),48)
        self.assertEqual(set(common.CONFIG["gpu_languages"].values()),set(common.CONFIG["languages"]))
        self.assertEqual(common.CONFIG["seeds"],[2028,2029,2030])

    def test_patience_and_restore(self):
        callback=EarlyStoppingCallback(10,0)
        args=SimpleNamespace(metric_for_best_model="macro_f1",greater_is_better=True)
        state=TrainerState(best_metric=0.8)
        for _ in range(9):
            control=TrainerControl()
            callback.on_evaluate(args,state,control,metrics={"eval_macro_f1":0.8})
        self.assertFalse(control.should_training_stop)
        restored=EarlyStoppingCallback.from_state(callback.state())
        self.assertEqual(restored.early_stopping_patience_counter,9)
        control=TrainerControl()
        restored.on_evaluate(args,state,control,metrics={"eval_macro_f1":0.79})
        self.assertTrue(control.should_training_stop)
        callback.on_evaluate(args,state,TrainerControl(),metrics={"eval_macro_f1":0.81})
        self.assertEqual(callback.early_stopping_patience_counter,0)

    def test_checkpoint_selection(self):
        with TemporaryDirectory(dir=common.ROOT/"tmp") as temp:
            root=Path(temp)
            for step in [10,20,30]:
                p=root/f"checkpoint-{step}";p.mkdir()
                for name in ["optimizer.pt","scheduler.pt","rng_state.pth","model.safetensors"]:
                    (p/name).write_text("fixture")
                (p/"trainer_state.json").write_text(json.dumps({"stateful_callbacks":{"EarlyStoppingCallback":{}}}))
            (root/"checkpoint-30"/"optimizer.pt").unlink()
            self.assertEqual(train.latest_complete_checkpoint(root),str(root/"checkpoint-20"))

    def test_summary(self):
        rows=[dict(language="en-US",model="xlmr",training_condition="clean",
                   test_condition="clean",typo_type="clean",noise_level=0,seed=seed,
                   accuracy=value,macro_f1=value,accuracy_drop=0,macro_f1_drop=0)
              for seed,value in zip([2028,2029,2030],[0.7,0.8,0.9])]
        result=export.summarize(pd.DataFrame(rows))
        self.assertAlmostEqual(result.iloc[0].macro_f1_mean,0.8)
        self.assertAlmostEqual(result.iloc[0].macro_f1_std,0.1)
        with self.assertRaises(AssertionError):export.summarize(pd.DataFrame(rows[:2]))
        with self.assertRaises(AssertionError):export.summarize(pd.DataFrame(rows+[rows[0]]))

    def test_gpu_busy_detection(self):
        with patch.object(runner.subprocess,"check_output",side_effect=["GPU-test","GPU-test, 123","2"]):
            with self.assertRaises(RuntimeError):runner.gpu_free(0)
        with patch.object(runner.subprocess,"check_output",side_effect=["GPU-test","","2"]):
            runner.gpu_free(0)
        with patch.object(runner.subprocess,"check_output",side_effect=["GPU-test","","500"]):
            with self.assertRaises(RuntimeError):runner.gpu_free(0)

    def test_cleanup_guard(self):
        with TemporaryDirectory(dir=common.ROOT/"tmp") as temp,patch.object(train,"ROOT",Path(temp)):
            root=Path(temp);checkpoint=root/"checkpoints"/"en__xlmr__clean__seed2028"
            checkpoint.mkdir(parents=True)
            train.cleanup_checkpoints(checkpoint,False);self.assertTrue(checkpoint.exists())
            with self.assertRaises(AssertionError):train.cleanup_checkpoints(root,True)
            train.cleanup_checkpoints(checkpoint,True);self.assertFalse(checkpoint.exists())

    def test_export_alignment(self):
        clean=pd.DataFrame({"source_row":[0,1],"intent":[0,1],"utterance":["alpha","beta"],"clean_utterance":["alpha","beta"]})
        noisy=clean.copy();noisy["utterance"]=["alhpa","btea"]
        sets=[("clean","clean",0,clean),("swap_0.1","swap",0.1,noisy)]
        with TemporaryDirectory(dir=common.ROOT/"tmp") as temp,patch.object(export,"ROOT",Path(temp)),patch.object(export,"test_sets",return_value=sets):
            result=export.export_run("en-US","xlmr","clean",2028,lambda texts:np.array([0,1]))
            frame=pd.read_csv(next((Path(temp)/"outputs/predictions").glob("*.csv")))
            self.assertEqual(result["prediction_rows"],4)
            self.assertEqual(frame.sample_id.tolist(),["en-US:test:0","en-US:test:1"]*2)
            self.assertEqual(frame.clean_utterance.tolist(),["alpha","beta"]*2)
            metrics=json.loads(next((Path(temp)/"outputs/test_metrics").glob("*.json")).read_text())
            self.assertEqual(metrics[1]["macro_f1_drop"],0)

    def test_detached_launcher(self):
        with TemporaryDirectory(dir=common.ROOT/"tmp") as temp,patch.object(runner,"ROOT",Path(temp)),patch.object(runner,"active_workers",return_value=[]),patch.object(runner,"gpu_free"),patch.object(runner,"check_disk"),patch.object(runner.time,"sleep"),patch.object(runner.subprocess,"Popen") as popen:
            popen.return_value.poll.return_value=None
            runner.start()
            kwargs=popen.call_args.kwargs
            self.assertTrue(kwargs["start_new_session"])
            self.assertTrue(kwargs["pass_fds"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"],"")

if __name__=="__main__":unittest.main()
