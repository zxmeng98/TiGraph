
import pandas as pd
import torch
import numpy as np
import time
import threading
import os

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy, TrainerCallback
from lm_workloads.lm_core.config import cfg, update_cfg
from lm_workloads.lm_core.model import BertClassifier, BertClaInfModel
from lm_workloads.lm_core.dataset import Dataset
from lm_workloads.lm_core.load import load_data
from lm_workloads.lm_core.utils import init_path, time_logger
from od_execution.od_execution import od_execution_wrapper
from od_execution.client import send_signal
from lm_workloads.lm_core.mytrainer import MyTrainer


def pause_notifier():
    import time
    time.sleep(15)
    import os
    print("send pause")
    pid = os.getpid()
    send_signal(pid, "pause")

def resume_notifier():
    import time
    time.sleep(24)
    import os
    print("send resume")
    pid = os.getpid()
    send_signal(pid, "resume")


def set_notifiers():
    t1 = threading.Thread(target=pause_notifier)
    t2 = threading.Thread(target=resume_notifier)
    t1.start()
    t2.start()

def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


pp_profile = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        skip_first=40, wait=0, warmup=0, active=20, repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        f"./tensorboard_trace/lm_worloads/"
    ),
    # f"./tensorboard_trace/revgnn_pp{num_stages}_stage{stage_index}_iter/"
    with_stack=True,
    with_modules=True,
    profile_memory=True,
)
        

class PrintEpochTimeCallback(TrainerCallback):
    """自定义回调：在每个 epoch 开始和结束时记录并打印耗时。"""

    def __init__(self):
        super().__init__()
        self.step_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        pp_profile.start()  # 开始 Profiler
        
    def on_step_begin(self, args, state, control, **kwargs):
        # 每次 iteration 开始时记录当前时间
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # 计算本次 iteration 的时长并打印
        step_time = time.time() - self.step_start_time
        pp_profile.step()  # 每步更新 Profiler 记录
        
        if state.log_history:
            loss = state.log_history[-1]["loss"] if "loss" in state.log_history[-1] else None
            print(f"Iter {state.global_step}/{state.max_steps}: Loss = {loss:.4f}, Iter Time = {step_time:.4f} s.")
        else:
            loss = None
        # print(f"Iter {state.global_step}/{state.max_steps}: Iter Time = {step_time:.4f} s.")
        
        
    def on_train_end(self, args, state, control, **kwargs):
        pp_profile.stop()  # 停止 Profiler
        print("Profiling Complete! View results in TensorBoard.")


class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        self.use_gpt_str = "2" if cfg.lm.train.use_gpt else ""
        self.output_dir = f'lm_workloads/output/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'
        self.ckpt_dir = f'prt_lm/{self.dataset_name}{self.use_gpt_str}/{self.model_name}-seed{self.seed}'

        # Preprocess data
        data, num_classes, text = load_data(
            dataset=self.dataset_name, use_text=True, use_gpt=cfg.lm.train.use_gpt, seed=self.seed)
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes

        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        X = tokenizer(text, padding=True, truncation=True, max_length=512) # tokenizer in 74.3060 seconds.
        t1 = time.time()
        print(f"tokenizer in {t1-t0:.4f} seconds.")

        dataset = Dataset(X, data.y.tolist())
        self.inf_dataset = dataset

        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name)
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)
        # prev_ckpt = f'prt_lm/{self.dataset_name}/{self.model_name}.ckpt'
        # if self.use_gpt_str and os.path.exists(prev_ckpt):
        #     print("Initialize using previous ckpt...")
        #     self.model.load_state_dict(torch.load(prev_ckpt))


        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

        # self.model = od_execution_wrapper(self.model)

    @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps) 

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=True, # drop last batch if not full
            disable_tqdm=True,
            # logging_dir="./logs",
            # logging_strategy="steps",
            logging_steps=1,
        )
        # self.trainer = Trainer(
        #     model=self.model,
        #     args=args,
        #     train_dataset=self.train_dataset,
        #     eval_dataset=self.val_dataset,
        #     compute_metrics=compute_metrics,
        #     # callbacks=[PrintEpochTimeCallback()],
        #     # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        # )
        self.trainer = MyTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[PrintEpochTimeCallback()],
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )


        # Train pre-trained model
        # train:valid:test = 90941:29799:48603
        # steps: len(train_dataset) // batch_size * epochs = 90941 // 9 = 10104

        # 从启动到这里大概2min左右
        # set_notifiers()
        self.trainer.train() 
        # torch.save(self.model.state_dict(), init_path(f"{self.ckpt_dir}.ckpt"))
        # print(f'LM saved to {self.ckpt_dir}.ckpt')

    @time_logger
    @torch.no_grad()
    def eval_and_save(self):
        emb = np.memmap(init_path(f"{self.ckpt_dir}.emb"),
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
        pred = np.memmap(init_path(f"{self.ckpt_dir}.pred"),
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)
        if "ogbn" in self.dataset_name:
            from ogb.nodeproppred import Evaluator
            _evaluator = Evaluator(name=self.dataset_name)
        # else:
        #     from core.GNNs.gnn_utils import Evaluator
        #     _evaluator = Evaluator(name=self.dataset_name)

        def evaluator(preds, labels): return _evaluator.eval({
            "y_true": torch.tensor(labels).view(-1, 1),
            "y_pred": torch.tensor(preds).view(-1, 1),
        })["acc"]

        def eval(x): return evaluator(
            np.argmax(pred[x], -1), self.data.y[x])

        train_acc = eval(self.data.train_mask)
        val_acc = eval(self.data.val_mask)
        test_acc = eval(self.data.test_mask)
        print(
            f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc}



def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    for seed in seeds:
        cfg.seed = seed
        trainer = LMTrainer(cfg)
        trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == '__main__':
    print(f"PID: {os.getpid()}")
    cfg = update_cfg(cfg)
    run(cfg)
