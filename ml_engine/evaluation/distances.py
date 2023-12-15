from typing import Dict

import pandas as pd
import torch
from torch import Tensor

from ml_engine.utils import get_combinations


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


def compute_distance_matrix_from_embeddings(embeddings, distance_fn, batch_size=256):
    """
    Compute the distance matrix between elements in embeddings
    @param embeddings: N x M tensor
    @param distance_fn: distance function
    @param batch_size: for reduce computing overheat
    @return: N x N distance matrix
    """
    size = len(embeddings)
    distance_matrix = torch.zeros((size, size), dtype=embeddings.dtype)
    combinations = get_combinations(torch.arange(size), torch.arange(size))
    all_scores = []
    for chunk in torch.split(combinations, batch_size):
        scores = distance_fn(embeddings[chunk[:, 0]], embeddings[chunk[:, 1]])
        all_scores.append(scores.cpu())

    scores = torch.cat(all_scores, dim=0)
    distance_matrix[combinations[:, 0], combinations[:, 1]] = scores
    distance_matrix[combinations[:, 1], combinations[:, 0]] = scores
    return distance_matrix
