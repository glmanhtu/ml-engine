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
