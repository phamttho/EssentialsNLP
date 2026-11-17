from .common import *
import subprocess

def validate():
    m=manifest()
    assert m["languages"] == CONFIG["languages"]
    assert sorted(label_mapping()) == list(range(60))
    report={"languages":{}, "data_sha256":{}, "warnings":[],
            "source_row_definition":"Zero-based row in the frozen exported clean split, not original MASSIVE ID"}
    for lang in CONFIG["languages"]:
        tr=clean(lang,"train")
        va=clean(lang,"validation")
        te=clean(lang,"test")
        assert set(tr.intent)==set(label_mapping()), f"Missing training labels: {lang}"
        aug=training(lang,"mixed")
        assert {"source_row","is_corrupted","typo_type","noise_level"} <= set(aug.columns)
        flag=aug.is_corrupted.astype(str).str.lower()
        assert set(flag) <= {"true","false","0","1"}
        is_noisy=flag.isin(["true","1"])
        assert len(aug)==2*len(tr), f"Expected 50/50 mixture for {lang}"
        for noisy in [False,True]:
            part=aug.loc[is_noisy == noisy].sort_values("source_row")
            assert part.source_row.tolist()==list(range(len(tr))), f"Training row mapping: {lang}"
            assert part.intent.tolist()==tr.intent.tolist()
            if not noisy:
                assert part.utterance.tolist()==tr.utterance.tolist(), f"Clean portion mismatch: {lang}"
        assert not tr.utterance.str.strip().eq("").any(), f"Empty clean training: {lang}"
        assert not va.utterance.str.strip().eq("").any()
        assert not te.utterance.str.strip().eq("").any()
        conditions=[]
        for name,typo,level,df in test_sets(lang):
            assert len(df)==len(te)
            assert sorted(df.source_row.tolist())==list(range(len(te))), f"Test IDs: {lang}/{name}"
            ordered=df.sort_values("source_row")
            assert ordered.intent.tolist()==te.intent.tolist()
            assert ordered.clean_utterance.tolist()==te.utterance.tolist()
            if name!="clean":
                assert df.language.eq(lang).all()
                assert df.split.eq("test").all()
                assert df.typo_type.eq(typo).all()
                assert np.allclose(pd.to_numeric(df.noise_level),level)
            conditions.append({"condition":name,"rows":len(df),
                               "unchanged":int(df.utterance.eq(df.clean_utterance).sum()),
                               "empty":int(df.utterance.str.strip().eq("").sum())})
        assert len(conditions)==13, f"Expected 13 conditions: {lang}"
        report["languages"][lang]={"train":len(tr),"validation":len(va),"test":len(te),
                                   "mixed":len(aug),"conditions":conditions}
    for path in sorted(DATA.rglob("*")):
        if path.is_file():
            report["data_sha256"][path.relative_to(DATA).as_posix()]=sha(path)
    report["repository_commit"]=subprocess.check_output(["git","-C",str(DATA.parent),"rev-parse","HEAD"],text=True).strip()
    report["split_note"]="Uses teammate exports as provided; no split modification. Export notebook may apply cleaned_data."
    save_json(ROOT/"outputs/manifests/data_check.json",report)
    print(json.dumps({"status":"PASS","commit":report["repository_commit"],
                      "files":len(report["data_sha256"]),
                      "rows":{k:{s:v[s] for s in ["train","validation","test","mixed"]} for k,v in report["languages"].items()}},indent=2))
    return report

if __name__=="__main__":
    validate()
