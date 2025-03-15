import torch
from . import BaseOpHook
import time
import os

class ExecutionOpHook(BaseOpHook):
    def __init__(self):
        super().__init__()
        self.state = 1

    def sleep_while_paused(self):
        while self.state == 0:
            # print('stuck pause')
            time.sleep(0.001)

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        # torch.cuda.synchronize()
        self.sleep_while_paused()
        # torch.cuda.synchronize()
        # print(f'FWD PRE {module.__class__.__name__}')

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        # torch.cuda.synchronize()
        self.sleep_while_paused()
        # torch.cuda.synchronize()
        # print(f'FWD POST {module.__class__.__name__}')

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        # torch.cuda.synchronize()
        self.sleep_while_paused()
        # torch.cuda.synchronize()
        # print(f'BWD PRE {module.__class__.__name__}')

    def post_bwd_exec(self, module: torch.nn.Module, input):
        # torch.cuda.synchronize()
        self.sleep_while_paused()
        # torch.cuda.synchronize()
        # print(f'BWD POST {module.__class__.__name__}')
    
    def pre_iter(self):
        pass

    def post_iter(self):
        print(f'post_iter')

    def pause(self):
        # print(f"{os.getpid()} ExecutionOpHook: pause")
        self.state = 0
    
    def resume(self):
        # print(f"{os.getpid()} ExecutionOpHook: resume")
        self.state = 1
