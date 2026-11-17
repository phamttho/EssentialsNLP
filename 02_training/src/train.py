from .common import *
import argparse
import time
import traceback
import warnings
import fcntl
import shutil

def latest_complete_checkpoint(directory):
    required=["trainer_state.json","optimizer.pt","scheduler.pt","rng_state.pth","model.safetensors"]
    valid=[]
    for path in directory.glob("checkpoint-*"):
        if not path.name.split("-")[-1].isdigit():continue
        if all((path/name).is_file() and (path/name).stat().st_size>0 for name in required):
            try:
                state=json.loads((path/"trainer_state.json").read_text())
                # Restoring the patience counter is required for equivalent resume.
                if "EarlyStoppingCallback" in state.get("stateful_callbacks",{}):
                    valid.append(path)
            except (ValueError,OSError):pass
    return str(max(valid,key=lambda p:int(p.name.split("-")[-1]))) if valid else None

def cleanup_checkpoints(directory,verified):
    if not verified or not CONFIG["cleanup_completed_checkpoints"]:return
    expected=ROOT/"checkpoints"
    assert directory.parent==expected and directory.resolve().is_relative_to(expected.resolve())
    assert directory.name.count("__")==3 and not directory.is_symlink()
    if directory.exists():shutil.rmtree(directory)

def train_tfidf(lang,model,condition,seed,directory):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import joblib
    path=directory/"model.joblib"
    marker=directory/"training_complete.json"
    if marker.exists():
        summary=json.loads(marker.read_text())
        assert model_hashes(directory)==summary["model_hashes"]
        pipeline=joblib.load(path)
    else:
        settings=dict(CONFIG[model]);settings["ngram_range"]=tuple(settings["ngram_range"])
        settings["dtype"]=np.float64
        if model=="char_tfidf":settings["analyzer"]="char"
        elif lang=="zh-CN":settings.update(tokenizer=chinese_tokens,token_pattern=None)
        pipeline=Pipeline([("tfidf",TfidfVectorizer(**settings)),
            ("classifier",LogisticRegression(**CONFIG["logistic_regression"],random_state=seed))])
        df=training(lang,condition)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always",ConvergenceWarning)
            pipeline.fit(df.utterance.tolist(),df.intent.to_numpy())
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):
            raise RuntimeError("Logistic Regression did not converge; no result marked complete")
        for warning in caught: print("WARNING",str(warning.message),flush=True)
        tmp=path.with_suffix(".joblib.tmp")
        joblib.dump(pipeline,tmp);tmp.replace(path)
        # Reload to check the actual saved artifact before reporting success.
        restored=joblib.load(path)
        assert np.array_equal(pipeline.predict(df.utterance.iloc[:16].tolist()),
                              restored.predict(df.utterance.iloc[:16].tolist()))
        summary={"rows":len(df),"converged":True,"iterations":int(pipeline["classifier"].n_iter_.max()),
                 "actual_epochs":None,"best_epoch":None,"stop_reason":"solver_converged",
                 "model_hashes":model_hashes(directory)}
        save_json(marker,summary)
    va=clean(lang,"validation")
    return pipeline.predict,metrics(va.intent,pipeline.predict(va.utterance.tolist())),summary

def train_xlmr(lang,condition,seed,directory,checkpoint_dir):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in CONFIG["gpu_languages"]:
        raise RuntimeError("Specify exactly one allowed physical GPU: 0, 1, 2 or 3")
    import torch
    from transformers import (AutoTokenizer,AutoModelForSequenceClassification,Trainer,
        TrainingArguments,DataCollatorWithPadding,set_seed,EarlyStoppingCallback)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(CONFIG["threads_per_worker"])
    set_seed(seed)
    cfg=CONFIG["xlmr"]
    info=json.loads((ROOT/"cache/pretrained/metadata.json").read_text())
    source=info["snapshot"]
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self,texts,labels=None):
            self.tokens=tokenizer(texts,truncation=True,max_length=cfg["max_length"])
            self.labels=labels
        def __len__(self):return len(self.tokens["input_ids"])
        def __getitem__(self,i):
            result={k:v[i] for k,v in self.tokens.items()}
            if self.labels is not None:result["labels"]=int(self.labels[i])
            return result
    marker=directory/"training_complete.json"
    complete=marker.exists()
    if complete:
        summary=json.loads(marker.read_text())
        assert model_hashes(directory)==summary["model_hashes"]
    ids=label_mapping()
    model=AutoModelForSequenceClassification.from_pretrained(
        str(directory) if complete else source,local_files_only=True,
        num_labels=60,id2label=ids,label2id={v:k for k,v in ids.items()},attn_implementation="sdpa")
    args=TrainingArguments(
        output_dir=str(checkpoint_dir),num_train_epochs=cfg["max_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=cfg["learning_rate"],weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="epoch",save_strategy="epoch",save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,metric_for_best_model="macro_f1",greater_is_better=True,
        bf16=cfg["precision"]=="bf16",fp16=cfg["precision"]=="fp16",
        seed=seed,data_seed=seed,logging_steps=cfg["logging_steps"],
        logging_dir=str(ROOT/"logs"/run_id(lang,"xlmr",condition,seed)),
        report_to=[],dataloader_num_workers=0,optim="adamw_torch",
        disable_tqdm=True,eval_accumulation_steps=8,restore_callback_states_from_checkpoint=True)
    va=clean(lang,"validation")
    trainer=Trainer(model=model,args=args,eval_dataset=TextDataset(va.utterance.tolist(),va.intent.tolist()),
        data_collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8),processing_class=tokenizer,
        compute_metrics=lambda p:metrics(p.label_ids,np.argmax(p.predictions,axis=-1)),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["patience"],
                                         early_stopping_threshold=cfg["min_delta"])])
    if not complete:
        df=training(lang,condition)
        trainer.train_dataset=TextDataset(df.utterance.tolist(),df.intent.tolist())
        resume=latest_complete_checkpoint(checkpoint_dir)
        if resume: print("RESUME",resume,flush=True)
        trainer.train(resume_from_checkpoint=resume)
        history=trainer.state.log_history
        best=Path(trainer.state.best_model_checkpoint)
        best_state=json.loads((best/"trainer_state.json").read_text())
        actual=float(trainer.state.epoch)
        patience_state=next(c for c in trainer.callback_handler.callbacks if isinstance(c,EarlyStoppingCallback))
        stop=("early_stopping" if actual<cfg["max_epochs"]-1e-6 else "epoch_limit")
        trainer.save_model(str(directory))
        tokenizer.save_pretrained(str(directory))
        # Ensure the exported weights really are the chosen validation checkpoint.
        assert sha(directory/"model.safetensors")==sha(best/"model.safetensors")
        summary={"rows":len(df),"actual_epochs":actual,"best_epoch":float(best_state["epoch"]),
            "global_step":trainer.state.global_step,"best_checkpoint":best.name,
            "best_validation_macro_f1":trainer.state.best_metric,"stop_reason":stop,
            "patience_counter_at_stop":patience_state.early_stopping_patience_counter,
            "resumed_from":resume,"model_hashes":model_hashes(directory)}
        save_json(directory/"training_history.json",history)
        save_json(marker,summary)
    val=trainer.evaluate()
    assert abs(val["eval_macro_f1"]-summary["best_validation_macro_f1"])<1e-9
    def predict(texts):
        return np.argmax(trainer.predict(TextDataset(texts)).predictions,axis=-1)
    return predict,{"accuracy":val["eval_accuracy"],"macro_f1":val["eval_macro_f1"]},summary

def run(lang,model,condition,seed,expected_fp):
    from .export import export_run,completed_valid
    args=(lang,model,condition,seed);rid=run_id(*args)
    mark=fingerprint()
    assert mark==expected_fp,"Config or data changed since launcher started"
    state_path=ROOT/"outputs/manifests"/f"{rid}.json"
    checkpoint_dir=ROOT/"checkpoints"/rid
    locks=ROOT/"locks";locks.mkdir(exist_ok=True)
    with open(locks/f"{rid}.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if completed_valid(args,mark):
            cleanup_checkpoints(checkpoint_dir,True)
            print("SKIP completed",rid,flush=True);return
        old=json.loads(state_path.read_text()) if state_path.exists() else {}
        state={"run_id":rid,"status":"running","fingerprint":mark,"pid":os.getpid(),
            "started":old.get("started",time.time()),"attempt_started":time.time(),
            "attempt":old.get("attempt",0)+1,"config":CONFIG,"seed":seed,
            "gpu":os.environ.get("CUDA_VISIBLE_DEVICES")}
        save_json(state_path,state)
        try:
            directory=model_dir(*args);directory.mkdir(parents=True,exist_ok=True)
            checkpoint_dir.mkdir(parents=True,exist_ok=True)
            if model=="xlmr":
                predictor,val,summary=train_xlmr(lang,condition,seed,directory,checkpoint_dir)
            else:predictor,val,summary=train_tfidf(lang,model,condition,seed,directory)
            save_json(ROOT/"outputs/validation_metrics"/f"{rid}.json",val)
            state.update(export_run(lang,model,condition,seed,predictor))
            state.update(status="completed",finished=time.time(),
                training_summary={k:v for k,v in summary.items() if k!="model_hashes"},
                model_hashes=summary["model_hashes"])
            save_json(state_path,state)
            assert completed_valid(args,mark)
            cleanup_checkpoints(checkpoint_dir,True)
        except BaseException:
            state.update(status="failed",finished=time.time(),error=traceback.format_exc())
            save_json(state_path,state)
            raise

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--language",choices=CONFIG["languages"],required=True)
    parser.add_argument("--model",choices=CONFIG["models"],required=True)
    parser.add_argument("--condition",choices=CONFIG["conditions"],required=True)
    parser.add_argument("--seed",type=int,choices=CONFIG["seeds"],required=True)
    parser.add_argument("--fingerprint",required=True)
    a=parser.parse_args()
    run(a.language,a.model,a.condition,a.seed,a.fingerprint)
