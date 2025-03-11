from .ophooks import *
import torch
import signal
import sys
import torch.nn as nn
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
all = ["od_execution_wrapper"]


signal_pause = signal.SIGUSR1
signal_resume = signal.SIGUSR2
global_engine = None

def pause_handler(sig, frame):
    global global_engine
    global_engine.pause()

def resume_handler(sig, frame):
    global global_engine
    global_engine.resume()

def register_signal_handler():
    signal.signal(signal_pause, pause_handler)
    signal.signal(signal_resume, resume_handler)

class Engine():
    def __init__(self, model, ophook_list):
        # super().__init__()
        self._ophook_list = ophook_list
        self._model = model
        register_signal_handler()
    
    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)
  
    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

    # def backward(self, loss):
    #     # loss.backward()
    #     for ophook in self._ophook_list:
    #         ophook.post_iter()

    def train(self):
        self._model.train()
    
    def pause(self):
        for ophook in self._ophook_list:
            if hasattr(ophook, "pause"):
                ophook.pause()

    def resume(self):
        for ophook in self._ophook_list:
            if hasattr(ophook, "resume"):
                ophook.resume()
    
    def save_results(self, filename):
        for ophook in self._ophook_list:
            ophook.save_results(filename)

    def zero_grad(self):
        self._model.zero_grad()

    def parameters(self):
        return self._model.parameters()

def od_execution_wrapper(model):
    global global_engine
    ophook_list = [ExecutionOpHook()]
    register_ophooks_recursively(model, ophook_list)
    engine = Engine(model, ophook_list)
    global_engine = engine
    return engine._model # NOTE!!! 这里还是要返回model，而不是engine实例，不然在LM Trainer中accelerator.accumulate(model) context会失效，每个step仍然会sync gradient。
    # 发现关键：accelerator.py L951：def no_sync(self, model)。如果model不是DDP类，context会失效。