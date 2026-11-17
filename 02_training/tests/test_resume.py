"""Synthetic CPU-only integration test, not a project experiment or pilot."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from src.common import ROOT,sha
from src.train import latest_complete_checkpoint
import torch
from transformers import BertConfig,BertForSequenceClassification,Trainer,TrainingArguments,EarlyStoppingCallback,TrainerCallback

class StopAtThree(TrainerCallback):
    def on_epoch_end(self,args,state,control,**kwargs):
        if state.epoch>=3:control.should_training_stop=True

class ResumeContract(unittest.TestCase):
    def test_real_trainer_resume(self):
        torch.set_num_threads(1)
        data=[{"input_ids":torch.tensor([1,2,3,4]),"attention_mask":torch.ones(4,dtype=torch.long),"labels":i%2} for i in range(4)]
        with TemporaryDirectory(dir=ROOT/"tmp") as temp:
            folder=Path(temp)
            def trainer(stop=False):
                model=BertForSequenceClassification(BertConfig(vocab_size=32,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,max_position_embeddings=16,num_labels=2))
                args=TrainingArguments(output_dir=str(folder/"checkpoints"),use_cpu=True,num_train_epochs=30,
                    per_device_train_batch_size=4,per_device_eval_batch_size=4,
                    eval_strategy="epoch",save_strategy="epoch",save_total_limit=2,
                    load_best_model_at_end=True,metric_for_best_model="macro_f1",greater_is_better=True,
                    restore_callback_states_from_checkpoint=True,report_to=[],disable_tqdm=True,
                    logging_strategy="no",seed=2028)
                callbacks=[EarlyStoppingCallback(10,0)]
                if stop:callbacks.append(StopAtThree())
                return Trainer(model=model,args=args,train_dataset=data,eval_dataset=data,
                    compute_metrics=lambda p:{"macro_f1":0.5},callbacks=callbacks)
            first=trainer(True);first.train()
            self.assertEqual(first.state.epoch,3)
            checkpoint=latest_complete_checkpoint(folder/"checkpoints")
            self.assertIsNotNone(checkpoint)
            second=trainer();second.train(resume_from_checkpoint=checkpoint)
            self.assertEqual(second.state.epoch,11)
            callback=next(c for c in second.callback_handler.callbacks if isinstance(c,EarlyStoppingCallback))
            self.assertEqual(callback.early_stopping_patience_counter,10)
            output=folder/"final";second.save_model(str(output))
            self.assertEqual(sha(output/"model.safetensors"),sha(Path(second.state.best_model_checkpoint)/"model.safetensors"))
