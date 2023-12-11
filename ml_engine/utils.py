# --------------------------------------------------------
# Adapted from https://github.com/microsoft/Swin-Transformer
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import collections
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from torch import inf


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def n_batches(size, current_batch=-1):
    total = []
    for i in range(size):
        if i == current_batch:
            return len(total)
        for j in range(size):
            if j < i:
                continue
            total.append(j)
    return len(total)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def split_list_by_ratios(lst, ratios):
    total_len = len(lst)
    split_points = [int(ratio * total_len) for ratio in ratios]
    sublists = []
    start_idx = 0

    for split_point in split_points:
        sublist = lst[start_idx:start_idx + split_point]
        sublists.append(sublist)
        start_idx += split_point

    # Add the remaining elements to the last sublist
    sublists[-1].extend(lst[start_idx:])

    return sublists


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_ddp_config():
    local_rank, rank, world_size = 0, 0, 1

    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])

    if 'LOCAL_RANK' in os.environ:  # for torch.distributed.launch
        local_rank = int(os.environ["LOCAL_RANK"])

    elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'

    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(random.randint(10000, 65000))

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    return local_rank, rank, world_size


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    results = []
    for i in range(0, n):
        chunk = l[i::n]
        if len(chunk) > 0:
            results.append(chunk)
    return results


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = collections.defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=int)
    return labels_to_indices


def get_combinations(tensor1, tensor2):
    """
    Get all combinations of the two given tensors
    @param tensor1: 1D tensor
    @param tensor2: 1D tensor
    @return: torch.Tensor
    """
    # Create a grid of all combinations
    grid_number, grid_vector = torch.meshgrid(tensor1, tensor2, indexing='ij')

    # Stack the grids to get all combinations
    return torch.stack((grid_number, grid_vector), dim=-1).reshape(-1, 2)


def extract_params_from_omegaconf_dict(params):
    result = {}
    for param_name, element in params.items():
        tmp = _explore_recursive(param_name, element)
        result = {**result, **tmp}
    return result


def _explore_recursive(parent_name, element):
    results = {}
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                tmp = _explore_recursive(f'{parent_name}.{k}', v)
                results = {**results, **tmp}
            else:
                results[f'{parent_name}.{k}'] = v
    elif isinstance(element, ListConfig):
        results[f'{parent_name}'] = str(element)

    return results