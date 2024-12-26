# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch import fx
import numpy as np


logger = logging.getLogger(__name__)


def flatten_args_detach(args):
    """
    Flatten the args into a list form and detach the tensors from computational graph.
    """
    flat_detached_args = []

    def extract_tensor_args(a):
        nonlocal flat_detached_args
        if isinstance(a, torch.Tensor):
            val = a.detach().requires_grad_(a.requires_grad)
            flat_detached_args.append(val)
            return val
        else:
            flat_detached_args.append(a)
            return a

    new_args = fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return new_args, flat_detached_args


def flatten_args(args):
    """
    Flatten the args into a list form.
    """
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        flat_args.append(a)
        return a

    fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return flat_args


class PipeliningShapeError(RuntimeError):
    """Shape mismatch between configured and runtime values."""


def validate_tensor_metadata(desc, expected, given):
    if not expected.shape == given.shape:
        raise PipeliningShapeError(
            f"{desc} has a shape mismatch: expected {expected.shape} actual {given.shape}"
        )
    if not expected.dtype == given.dtype:
        raise PipeliningShapeError(
            f"{desc} has a dtype mismatch: expected {expected.dtype} actual {given.dtype}"
        )
    if not expected.stride() == given.stride():
        raise PipeliningShapeError(
            f"{desc} has a stride mismatch: expected {expected.stride()} actual {given.stride()}"
        )


def validate_tensors_metadata(
    desc,
    expected_tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    actual_tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
):
    if len(expected_tensors) != len(actual_tensors):
        raise PipeliningShapeError(
            f"{desc}: Number of values ({len(actual_tensors)}) does not match expected number ({len(expected_tensors)})"
        )
    for i in range(len(expected_tensors)):
        validate_tensor_metadata(
            f"{desc}: value {i}", expected_tensors[i], actual_tensors[i]
        )


@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool


def partition_uniform(num_items, num_parts):
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = num_items // num_parts
    residual = num_items - (chunksize * num_parts)

    parts = np.arange(0, (num_parts + 1) * chunksize, chunksize)

    for i in range(residual):
        parts[i + 1:] += 1
    parts = parts.tolist()

    return parts


def partition_balanced(weights, num_parts):
    """
    use dynamic programming solve `The Linear Partition Problem`.
    see https://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK2/NODE45.HTM
    """
    import numpy as np
    n = len(weights)
    m = num_parts

    if n <= m:
        return partition_uniform(n, m)

    dp_max = np.full((n + 1, m + 1), np.inf)
    dp_min = np.full((n + 1, m + 1), np.inf)
    dp_cost = np.full((n + 1, m + 1), np.inf)
    position = np.zeros((n + 1, m + 1), dtype=int)
    prefix_sum = np.zeros((n + 1))
    prefix_sum[1:] = np.cumsum(weights)

    dp_max[0, 0] = 0
    dp_cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, min(i, m) + 1):
            for k in range(i):
                max_sum = max(dp_max[k, j - 1], prefix_sum[i] - prefix_sum[k])
                min_sum = min(dp_min[k, j - 1], prefix_sum[i] - prefix_sum[k])
                cost = max_sum - min_sum
                if dp_cost[i, j] >= cost:
                    dp_cost[i, j] = cost
                    dp_max[i, j] = max_sum
                    dp_min[i, j] = min_sum
                    position[i, j] = k

    parts = [n]
    for i in reversed(range(1, m + 1)):
        parts.append(position[parts[-1], i])
    parts.reverse()

    return parts