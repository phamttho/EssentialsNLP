"""Pre-deployment self-check; no scientific training or test-based selection."""
from base import *
import unittest,contextlib,platform
def main():
    logfile=ROOT/"state/selfcheck.log"
    with logfile.open("w") as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        suite=unittest.defaultTestLoader.loadTestsFromNames(["tests_runner","tests_eval"])
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        assert result.wasSuccessful()
        # Re-import one previously generated file, not new test inference.
        from evaluate import import_legacy
        legacy=import_legacy("en-US","word_tfidf","clean",2028,1.,ROOT/"state/legacy_import_selfcheck")
        from fit import smoke_test
        toy=smoke_test(ROOT/"work"/"smoke_training_callback")
    selfchecks={"unit_tests_passed":result.testsRun,"real_cpu_trainer":toy,"legacy_import":legacy,
      "hardware":json.loads((ROOT/"state/hardware_check.json").read_text()),
      "status":"PASS","production_training_started":False}
    save_json(ROOT/"state/selfcheck.json",selfchecks)
    # Only our three named synthetic temporary model directories.
    for name in ("smoke_training_callback","smoke_word_tfidf","smoke_char_tfidf"):
        p=ROOT/"work"/name
        if p.exists():
            history=p/"training_history.json"
            if history.exists():shutil.copy2(history,ROOT/"state"/(name+"_history.json"))
            clear_work(p)
    # Explicit patch leftovers created during deployment, not user data.
    for name in ("evaluate.py.rej","tests_eval.py.rej","tests_eval.py.orig"):
        p=ROOT/name
        if p.exists():
            assert p.is_file() and not p.is_symlink() and p.resolve().parent==ROOT
            p.unlink()
    from run import preflight
    report=preflight()
    save_json(ROOT/"state/deployment.json",{"status":"READY_NOT_STARTED","fingerprint":experiment_fingerprint(),
      "preflight":report,"selfchecks":selfchecks,"files":{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.glob("*.py"))}})
    assert not (ROOT/"state/runner.json").exists(),"Unexpected production runner state"
    assert not list((ROOT/"records").rglob("*.json")),"Unexpected production records"
    assert not list((ROOT/"work").iterdir()),"Unexpected temporary weight directories"
    print(json.dumps({"status":"READY_NOT_STARTED","tests":result.testsRun,"disk_free_gib":report["free_gib"],"gpus":report["gpus"],"temporary_weight_directories":0},indent=2))
if __name__=="__main__":main()
