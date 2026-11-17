"""Synthetic export tests. No pretrained model or production prediction is evaluated."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import evaluate as ev

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.parent=ev.ROOT/"selftest"
        self.parent.mkdir(exist_ok=True)
        self.tmp=TemporaryDirectory(prefix="evaluation_",dir=self.parent)
        self.root=Path(self.tmp.name)
        self.dest=self.root/"output"
        self.args=("en-US","xlmr","clean",2028,2e-5)
        self.patch=patch.object(ev,"test_sets",self.source_sets)
        self.patch.start()
        ev.export_and_verify(*self.args,lambda texts:np.zeros(len(texts),dtype=int),self.dest)
    def tearDown(self):
        self.patch.stop()
        assert self.root.resolve().is_relative_to((ev.ROOT/"selftest").resolve())
        self.tmp.cleanup()
    @staticmethod
    def source_sets(lang):
        n=ev.EXPECTED_TEST_ROWS
        df=pd.DataFrame({"utterance":[f"test sample {i}" for i in range(n)],
                         "intent":np.arange(n)%60,"source_row":np.arange(n)})
        df["clean_utterance"]=df.utterance
        yield "clean","none",0.,df.copy()
        for typo in ("deletion","substitution","duplication","swap"):
            for lev in (.05,.10,.20):
                noisy=df.copy()
                noisy.loc[0,"utterance"]=""
                yield f"{typo}_{lev:.2f}",typo,lev,noisy
    def frame(self):
        return pd.read_csv(self.dest/"predictions.csv.gz",keep_default_na=False)
    def write_frame(self,frame):
        frame.to_csv(self.dest/"predictions.csv.gz",index=False,compression="gzip")
    def test_export_and_reload(self):
        result=ev.verify_existing(*self.args,self.dest)
        self.assertEqual(result["prediction_rows"],38662)
        self.assertEqual(result["test_conditions"],13)
        self.assertEqual((self.frame().test_utterance=="").sum(),12)
    def test_changed_source_text_rejected(self):
        frame=self.frame();frame.loc[0,"clean_utterance"]="tampered"
        self.write_frame(frame)
        with self.assertRaises(AssertionError):ev.verify_existing(*self.args,self.dest,check_marker=False)
    def test_changed_true_label_rejected(self):
        frame=self.frame();frame.loc[0,"true_intent"]=7
        self.write_frame(frame)
        with self.assertRaises(AssertionError):ev.verify_existing(*self.args,self.dest,check_marker=False)
    def test_duplicate_key_rejected(self):
        frame=self.frame();frame.loc[1]=frame.loc[0]
        self.write_frame(frame)
        with self.assertRaises(AssertionError):ev.verify_existing(*self.args,self.dest,check_marker=False)
    def test_changed_metric_rejected(self):
        rows=json.loads((self.dest/"metrics.json").read_text())
        rows[0]["macro_f1"]=.999
        ev.save_json(self.dest/"metrics.json",rows)
        with self.assertRaises(AssertionError):ev.verify_existing(*self.args,self.dest,check_marker=False)
    def test_invalid_prediction_rejected(self):
        frame=self.frame();frame.loc[0,"predicted_intent"]=60
        self.write_frame(frame)
        with self.assertRaises(AssertionError):ev.verify_existing(*self.args,self.dest,check_marker=False)
    def test_stale_hash_marker_rejected(self):
        rows=json.loads((self.dest/"verification.json").read_text())
        rows["predictions_sha256"]="0"*64
        ev.save_json(self.dest/"verification.json",rows)
        with self.assertRaises(AssertionError):ev.verify_existing(*self.args,self.dest)
    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):ev._destination(ev.ROOT.parent/"must_not_write")
    def test_wrong_legacy_candidate_rejected(self):
        with self.assertRaises(AssertionError):ev.import_legacy("en-US","xlmr","clean",2028,3e-5,self.dest)
    def test_missing_final_runs_rejected(self):
        with self.assertRaises(AssertionError):ev.aggregate([])
    def test_legacy_import_checks_hashes_and_values(self):
        old=self.root/"old_fixture"
        for part in ("predictions","test_metrics","manifests"):
            (old/"outputs"/part).mkdir(parents=True)
        rid="en-US__xlmr__clean__seed2028"
        pp=old/"outputs/predictions"/(rid+".csv")
        mp=old/"outputs/test_metrics"/(rid+".json")
        self.frame()[ev.BASE_COLUMNS].to_csv(pp,index=False)
        rows=json.loads((self.dest/"metrics.json").read_text())
        for row in rows:
            row.pop("parameter_name");row.pop("parameter_value")
        ev.save_json(mp,rows)
        ev.save_json(old/"outputs/manifests"/(rid+".json"),{
            "status":"completed","run_id":rid,"fingerprint":"synthetic",
            "config":{"xlmr":{"learning_rate":2e-5}},
            "predictions_sha256":ev.sha(pp),"metrics_sha256":ev.sha(mp)})
        with patch.object(ev,"OLD",old):
            result=ev.import_legacy(*self.args,self.root/"imported")
            self.assertTrue(result["verified"])
            self.assertEqual(result["prediction_rows"],38662)
            pp.write_text("tampered",encoding="utf-8")
            with self.assertRaises(AssertionError):
                ev.import_legacy(*self.args,self.root/"tampered_import")

    def test_full_aggregate_and_paired_delta(self):
        import itertools
        root=self.root/"aggregate_fixture"
        old=self.root/"baseline_fixture"
        (old/"outputs/test_metrics").mkdir(parents=True)
        (old/"outputs/manifests").mkdir(parents=True)
        (root/"records/final").mkdir(parents=True)
        prototype=json.loads((self.dest/"metrics.json").read_text())
        entries=[]
        for lang,model,cond,seed in itertools.product(ev.LANGUAGES,ev.MODELS,ev.CONDITIONS,ev.SEEDS):
            value=1e-5 if model=="xlmr" else .1
            rid=f"{lang}__{model}__{cond}__seed{seed}"
            cid=rid+"__value"+format(value,".8g")
            dest=root/"results/final"/cid;dest.mkdir(parents=True)
            original=json.loads(json.dumps(prototype))
            tuned=json.loads(json.dumps(prototype))
            info={"language":lang,"model":model,"training_condition":cond,"seed":seed,
                  "parameter_name":"learning_rate" if model=="xlmr" else "C","parameter_value":value}
            for rows in (original,tuned):
                for row in rows:row.update(info)
            for row in original:
                row["parameter_value"]=2e-5 if model=="xlmr" else 1.
            for row in tuned:
                row["accuracy"]+=(seed-2027)*.01
                row["macro_f1"]+=(seed-2027)*.01
            mp=old/"outputs/test_metrics"/(rid+".json")
            ev.save_json(mp,original)
            ev.save_json(old/"outputs/manifests"/(rid+".json"),{
                "status":"completed","metrics_sha256":ev.sha(mp)})
            ev.save_json(dest/"metrics.json",tuned)
            ev.save_json(root/"records/final"/(cid+".json"),{
                "status":"completed","job":[lang,model,cond,seed,value],"started":10.,"finished":70.,
                "provenance":"selected_parameter_retrained",
                "search_macro_f1":.7,"refit_validation_difference":.01,
                "summary":{"accuracy":.8,"macro_f1":.71,"rows":100,"actual_epochs":22,
                           "best_epoch":12,"stop_reason":"early_stopping"}})
            entries.append({"language":lang,"model":model,"condition":cond,"seed":seed,
                            "value":value,"dest":dest})
        with patch.object(ev,"ROOT",root),patch.object(ev,"OLD",old),patch.object(ev,"verify_existing"):
            result=ev.aggregate(entries)
        self.assertEqual(result["evaluation_rows"],936)
        directory=root/"results/summaries"
        comparison=pd.read_csv(directory/"baseline_vs_tuned.csv")
        self.assertEqual(len(comparison),312)
        self.assertTrue(np.allclose(comparison.macro_f1_delta_mean,.02))
        self.assertTrue(np.allclose(comparison.macro_f1_delta_std,.01))
        training=pd.read_csv(directory/"final_training_summary.csv")
        self.assertEqual(len(training),72)
        self.assertTrue((training.best_epoch==12).all())
        self.assertTrue((training.final_stage_wall_minutes==1).all())

if __name__=="__main__":
    unittest.main(verbosity=2)
