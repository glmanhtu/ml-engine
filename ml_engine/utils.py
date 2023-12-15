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


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return


def revert_sync_batchnorm(module):
    # Since SyncBatchNorm doesn't work with torch.jit.script, we need to convert it back to normal batchnorm
    # See: https://github.com/pytorch/pytorch/issues/41081
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = BatchNormXd(module.num_features,
                                    module.eps, module.momentum,
                                    module.affine,
                                    module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


def compute_distance_matrix(embeddings, distance_fn):
    """
    Compute the distance matrix between elements in embeddings
    @param embeddings: N x M tensor
    @param distance_fn: distance function
    @return: N x N distance matrix
    """
    size = len(embeddings)
    distance_matrix = torch.zeros((size, size), dtype=torch.float32)
    combinations = get_combinations(torch.arange(size), torch.arange(size))
    scores = distance_fn(embeddings[combinations[:, 0]], embeddings[combinations[:, 1]])
    distance_matrix[combinations[:, 0], combinations[:, 1]] = scores
    distance_matrix[combinations[:, 1], combinations[:, 0]] = scores
    return distance_matrix
