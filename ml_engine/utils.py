# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
import os
import random
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import inf, Tensor


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    min_loss = 99999
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler and config.TRAIN.LOAD_LR_SCHEDULER:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'min_loss' in checkpoint:
            min_loss = checkpoint['min_loss']

    del checkpoint
    torch.cuda.empty_cache()
    return min_loss


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


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


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


class CalTimer:
    def __init__(self):
        self.functions = {}
        self.ordered = []
        self.current_time = None

    def time_me(self, func_name, current_time):
        time_diff = current_time - self.current_time
        self.current_time = current_time
        if func_name not in self.functions:
            self.functions[func_name] = AverageMeter()
            self.ordered.append(func_name)
        self.functions[func_name].update(time_diff)

    def set_timer(self):
        self.current_time = time.time()

    def get_results(self):
        results = ""
        for key in self.ordered:
            results += f'{key}: {self.functions[key].avg:.4f}\t'
        return results


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count


class UnableToCrop(Exception):
    def __init__(self, message, im_path=''):
        super().__init__(message + ' ' + im_path)
        self.im_path = im_path


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_ddp():
    local_rank, rank, world_size = -1, -1, -1

    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])

    if 'LOCAL_RANK' in os.environ:  # for torch.distributed.launch
        local_rank = int(os.environ["LOCAL_RANK"])

    elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = rank % torch.cuda.device_count()

    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    return local_rank, rank, world_size


def list_to_idx(items, name_converting_fn):
    labels = [name_converting_fn(x) for x in items]
    authors = list(set(labels))
    author_map = {x: i for i, x in enumerate(authors)}
    labels = [author_map[x] for x in labels]
    return labels


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    results = []
    for i in range(0, n):
        chunk = l[i::n]
        if len(chunk) > 0:
            results.append(chunk)
    return results


def get_repeated_indexes(input_size, output_size):
    n_times = math.ceil(output_size / input_size)
    indexes = [torch.arange(input_size) for _ in range(n_times)]
    res = torch.cat(indexes, dim=0)
    return res[torch.randperm(res.shape[0])][:output_size]


def get_combinations(tensor1, tensor2):
    # Create a grid of all combinations
    grid_number, grid_vector = torch.meshgrid(tensor1, tensor2, indexing='ij')

    # Stack the grids to get all combinations
    return torch.stack((grid_number, grid_vector), dim=-1).reshape(-1, 2)


def cosine_distance(source, target):
    similarity_fn = torch.nn.CosineSimilarity(dim=1)
    similarity = similarity_fn(source, target)
    return 1 - similarity


def compute_distance_matrix(data: Dict[str, Tensor], reduction='mean', distance_fn=cosine_distance):
    distance_map = {}
    fragments = list(data.keys())
    for i in range(len(fragments)):
        for j in range(i, len(fragments)):
            source, target = fragments[i], fragments[j]
            combinations = get_combinations(torch.arange(len(data[source])), torch.arange(len(data[target])))
            source_features = data[source][combinations[:, 0]]
            target_features = data[target][combinations[:, 1]]

            distance = distance_fn(source_features, target_features)
            if reduction == 'mean':
                distance = distance.mean()
            elif reduction == 'max':
                distance = torch.max(distance)
            elif reduction == 'min':
                distance = torch.min(distance)
            else:
                raise NotImplementedError(f"Reduction {reduction} is not implemented!")
            distance = distance.cpu().item()
            distance_map.setdefault(source, {})[target] = distance
            distance_map.setdefault(target, {})[source] = distance

    matrix = pd.DataFrame.from_dict(distance_map, orient='index').sort_index()
    return matrix.reindex(sorted(matrix.columns), axis=1)
