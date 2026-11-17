import unittest,tempfile,json,shutil
from pathlib import Path
from unittest.mock import patch
from collections import namedtuple
import base,run
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.dir=Path(tempfile.mkdtemp(prefix="unit_",dir=base.ROOT/"tmp"))
    def tearDown(self):shutil.rmtree(self.dir)
    def test_matrix(self):
        jobs=run.all_search_jobs()
        self.assertEqual(len(jobs),144)
        self.assertEqual(sum(x[1]=="xlmr" for x in jobs),48)
        self.assertEqual(len(set(jobs)),144)
    def test_selection_means_and_ties(self):
        (self.dir/"results").mkdir()
        for l in base.LANGUAGES:
            for m in base.MODELS:
                for c in base.CONDITIONS:
                    for s in base.SEEDS:
                        for v in base.GRID[m]:
                            # old is preferred for exact ties; non-default new value wins one group
                            score=.5
                            if l=="en-US" and m=="xlmr" and v==3e-5:score=.7 if c=="mixed" else .4
                            dest=self.dir/"records/search"/(base.cid(l,m,c,s,v)+".json")
                            base.save_json(dest,{"status":"completed","fingerprint":"unit","summary":{"macro_f1":score},"files":{},"provenance":"synthetic"})
        with patch.object(run,"ROOT",self.dir),patch.object(run,"experiment_fingerprint",return_value="unit"),patch.object(run,"assert_locked"):
            choices=run.select_parameters()
            self.assertEqual(len(choices),12)
            for x in choices:
                expected=3e-5 if x["language"]=="en-US" and x["model"]=="xlmr" else base.ORIGINAL[x["model"]]
                self.assertEqual(x["value"],expected)
            # Source contains no test scores; final gate accepts selected values only.
            run.assert_selected(("en-US","xlmr","mixed",2028,3e-5))
            with self.assertRaises(AssertionError):run.assert_selected(("en-US","xlmr","clean",2028,1e-5))
    def test_test_phase_requires_selection(self):
        with patch.object(run,"ROOT",self.dir):
            with self.assertRaises(AssertionError):run.assert_selected(("en-US","xlmr","clean",2028,1e-5))
    def test_missing_or_changed_lock(self):
        with patch.object(run,"ROOT",self.dir),patch.object(run,"experiment_fingerprint",return_value="actual"):
            with self.assertRaises(AssertionError):run.assert_locked()
            base.save_json(self.dir/"state/experiment.json",{"fingerprint":"wrong"})
            with self.assertRaises(AssertionError):run.assert_locked()
    def test_corrupted_completed_record_rejected(self):
        job=("en-US","word_tfidf","clean",2028,.1)
        out=self.dir/"out.txt";out.write_text("a")
        rec=self.dir/"records/search"/(base.cid(*job)+".json")
        base.save_json(rec,{"status":"completed","fingerprint":"unit","files":{"out.txt":base.sha(out)}})
        with patch.object(run,"ROOT",self.dir),patch.object(run,"experiment_fingerprint",return_value="unit"):
            self.assertTrue(run.valid_record("search",job))
            out.write_text("changed")
            with self.assertRaises(AssertionError):run.valid_record("search",job)
    def test_cleanup_refuses_old_and_parent(self):
        for path in (base.OLD/"models",base.ROOT/"work",base.ROOT/"outputs"):
            with self.assertRaises(AssertionError):base.clear_work(path)
    def test_disk_floor(self):
        usage=namedtuple("Usage","total used free")
        with patch.object(base.shutil,"disk_usage",return_value=usage(100,80,24*1024**3)):
            with self.assertRaises(RuntimeError):base.guard()
    def test_wrong_fingerprint_job_does_not_start(self):
        with patch.object(run,"ROOT",self.dir):
            with self.assertRaises(AssertionError):run.process_job("search",("en-US","xlmr","clean",2028,1e-5))
if __name__=="__main__":unittest.main(verbosity=2)
