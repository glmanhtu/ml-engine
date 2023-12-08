import torch
from torch import distributed as dist
import numpy as np


def calc_map_prak(distances, labels, positive_pairs, negative_pairs=None, prak=(1, 5)):
    """
    Calculate the mAP and Pr@K metrics from distance matrix according to the paper:
    Seuret, Mathias, et al. "ICFHR 2020 competition on image retrieval for historical handwritten fragments."
    2020 17th International conference on frontiers in handwriting recognition (ICFHR). IEEE, 2020

    @param distances: numpy distance matrix
    @param labels: 1D numpy or pd.Series labels
    @param positive_pairs: Dict[str, Set[str]] positive pairs
    @param negative_pairs: Dict[str, Set[str]] negative pairs (Optional)
    @param prak: tuple of Pr@K
    @return: mAP and Pr@K
    """

    avg_precision = []
    prak_res = [[] for _ in prak]

    for i in range(0, len(distances)):
        cur_dists = distances[i, :]
        idxs = np.argsort(cur_dists).flatten()
        sorted_labels = labels[idxs].tolist()
        pos_labels = positive_pairs[labels[i]]
        if negative_pairs is not None:
            neg_labels = negative_pairs[labels[i]]
            for li, label in reversed(list(enumerate(sorted_labels))):
                if label not in pos_labels and label not in neg_labels:
                    del sorted_labels[li]

        cur_sum = []
        pos_count = 1
        correct_count = []
        for idx, label in enumerate(sorted_labels):
            if idx == 0:
                continue    # First img is original image
            if label in pos_labels:
                cur_sum.append(float(pos_count) / idx)
                pos_count += 1
                correct_count.append(1)
            else:
                correct_count.append(0)

        if sum(correct_count) == 0:
            # If there is no positive pair, there should be a problem in GT
            # Ignore for now
            continue

        for i, k in enumerate(prak):
            val = sum(correct_count[:k]) / min(sum(correct_count), k)
            prak_res[i].append(val)


        ap = sum(cur_sum) / len(cur_sum)
        avg_precision.append(ap)


    m_ap = sum(avg_precision) / len(avg_precision)
    for i, k in enumerate(prak):
        prak_res[i] = sum(prak_res[i]) / len(prak_res[i])

    return m_ap, tuple(prak_res)


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

    @staticmethod
    def reduces(*meters):
        for meter in meters:
            meter.all_reduce()
