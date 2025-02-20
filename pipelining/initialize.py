import random
import os
import time

import numpy as np
import torch
from datetime import timedelta

import torch.distributed

# For pipeline parallel
_PIPELINE_PARALLEL_GROUP = None
_PIPELINE_PARALLEL_WORLD_SIZE = None
_PIPELINE_PARALLEL_RANK = None
_PIPELINE_PARALLEL_STAGE_INDEX = None
_PIPELINE_PARALLEL_NUM_STAGES = None

_SMALL_WORKLOAD_PID = None


def init_distributed():
    """Initialize torch.distributed and core model parallel."""
    global rank, device, pp_group, stage_index, num_stages
    device_count = torch.cuda.device_count()
    assert device_count != 0, 'expected GPU number > 0.'
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if rank == 0:
            print('> initializing torch distributed ...', flush=True)

        # Manually set the device ids.
        if device_count > 0:
            device = rank % device_count
            torch.cuda.set_device(device) # only do so when device_count > 0
        # Call the init process
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size, rank=rank,
        )

    global _PIPELINE_PARALLEL_GROUP
    global _PIPELINE_PARALLEL_WORLD_SIZE
    global _PIPELINE_PARALLEL_RANK
    global _PIPELINE_PARALLEL_STAGE_INDEX
    global _PIPELINE_PARALLEL_NUM_STAGES

    device = f'cuda:{torch.cuda.current_device()}' 
    _PIPELINE_PARALLEL_GROUP = torch.distributed.new_group()
    _PIPELINE_PARALLEL_WORLD_SIZE = world_size
    _PIPELINE_PARALLEL_RANK = rank
    _PIPELINE_PARALLEL_STAGE_INDEX = _PIPELINE_PARALLEL_RANK
    _PIPELINE_PARALLEL_NUM_STAGES = _PIPELINE_PARALLEL_WORLD_SIZE


def init_small_workload_pid(small_workload_pid):
    # Initialize small worload pid
    if small_workload_pid:
        global _SMALL_WORKLOAD_PID
        _SMALL_WORKLOAD_PID = small_workload_pid


def get_small_workload_pid():
    """Return small workload pid."""
    # assert _SMALL_WORKLOAD_PID is not None, \
    #     'small workload pid is not initialized'
    return _SMALL_WORKLOAD_PID
