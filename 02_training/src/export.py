from .common import *

def export_run(lang, model, condition, seed, predict):
    rid=run_id(lang,model,condition,seed)
    frames=[]; rows=[]; ids=label_mapping()
    for name,typo,level,df in test_sets(lang):
        pred=np.asarray(predict(df.utterance.tolist()),dtype=int)
        assert len(pred)==len(df) and set(pred)<=set(ids)
        info={"language":lang,"model":model,"training_condition":condition,
              "test_condition":name,"typo_type":typo,"noise_level":level,"seed":seed}
        out=pd.DataFrame({"source_row":df.source_row.to_numpy(),
            "sample_id":[f"{lang}:test:{i}" for i in df.source_row],
            "true_intent":df.intent.to_numpy(),"predicted_intent":pred,
            "true_intent_name":[ids[i] for i in df.intent],
            "predicted_intent_name":[ids[i] for i in pred],
            "clean_utterance":df.clean_utterance.to_numpy(),"test_utterance":df.utterance.to_numpy()})
        for k,v in info.items():out[k]=v
        frames.append(out)
        rows.append({**info,"n_samples":len(df),**metrics(df.intent,pred)})
        print("TEST",rid,name,rows[-1]["macro_f1"],flush=True)
    base=rows[0]
    for row in rows:
        row["accuracy_drop"]=base["accuracy"]-row["accuracy"]
        row["macro_f1_drop"]=base["macro_f1"]-row["macro_f1"]
    dest=ROOT/"outputs/predictions"/f"{rid}.csv"
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(".csv.tmp")
    pd.concat(frames,ignore_index=True).to_csv(temp,index=False)
    temp.replace(dest)
    met=ROOT/"outputs/test_metrics"/f"{rid}.json"
    save_json(met,rows)
    return {"prediction_rows":sum(map(len,frames)),"test_conditions":len(rows),
            "predictions_sha256":sha(dest),"metrics_sha256":sha(met)}

def completed_valid(args,mark,check_model=True):
    rid=run_id(*args)
    path=ROOT/"outputs/manifests"/f"{rid}.json"
    if not path.exists():return False
    state=json.loads(path.read_text())
    if state.get("fingerprint")!=mark:
        raise RuntimeError(f"Config/data/code changed for {rid}; refusing to mix experiments")
    if state.get("status")!="completed":return False
    assert sha(ROOT/"outputs/predictions"/f"{rid}.csv")==state["predictions_sha256"],rid
    assert sha(ROOT/"outputs/test_metrics"/f"{rid}.json")==state["metrics_sha256"],rid
    if check_model:
        assert model_hashes(model_dir(*args))==state["model_hashes"],rid
    return True

def summarize(frame):
    keys=["language","model","training_condition","test_condition","typo_type","noise_level"]
    values=["accuracy","macro_f1","accuracy_drop","macro_f1_drop"]
    for _,g in frame.groupby(keys,dropna=False):
        assert sorted(g.seed.tolist())==sorted(CONFIG["seeds"]), "Missing/duplicated seed"
    grouped=frame.groupby(keys,dropna=False)[values].agg(["mean","std"])
    grouped.columns=["_".join(c) for c in grouped.columns]
    result=grouped.reset_index()
    result["n_seeds"]=len(CONFIG["seeds"])
    return result

def aggregate():
    mark=fingerprint(); rows=[]; training_rows=[]
    for args in all_runs():
        rid=run_id(*args)
        assert completed_valid(args,mark),rid
        state=json.loads((ROOT/"outputs/manifests"/f"{rid}.json").read_text())
        part=json.loads((ROOT/"outputs/test_metrics"/f"{rid}.json").read_text())
        assert len(part)==13
        rows.extend(part)
        training_rows.append({"run_id":rid,"language":args[0],"model":args[1],
            "training_condition":args[2],"seed":args[3],
            "minutes":(state["finished"]-state["started"])/60,
            **state["training_summary"]})
    frame=pd.DataFrame(rows)
    assert len(frame)==936
    folder=ROOT/"outputs/summaries"
    folder.mkdir(parents=True,exist_ok=True)
    for name,df in [("all_test_metrics.csv",frame),("three_seed_summary.csv",summarize(frame)),
                    ("training_summary.csv",pd.DataFrame(training_rows))]:
        path=folder/name
        temp=path.with_suffix(".csv.tmp")
        df.to_csv(temp,index=False)
        temp.replace(path)
    save_json(ROOT/"outputs/manifests/handoff.json",{
        "status":"completed","runs":72,"evaluation_rows":936,"summary_rows":312,
        "seeds":CONFIG["seeds"],"fingerprint":mark,"std_ddof":1,
        "metric_scale":"0 to 1; drop = clean minus noisy per seed, then mean/std",
        "baseline":"word_tfidf / clean","source_row":"zero-based frozen exported clean test CSV row",
        "noise_data":"fixed across all training seeds"})
    print("COMPLETE: 72 models, 936 evaluations, 312 three-seed summary rows",flush=True)

if __name__=="__main__":aggregate()
