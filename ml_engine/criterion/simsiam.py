import torch

from ml_engine.criterion.losses import NegativeCosineSimilarityLoss
from ml_engine.utils import get_combinations


class BatchWiseSimSiamLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = NegativeCosineSimilarityLoss()

    def forward(self, embeddings, targets):
        ps, zs = embeddings
        return self.forward_impl(ps, zs, targets)

    def forward_impl(self, ps, zs, targets):
        n = ps.size(0)
        eyes_ = torch.eye(n, dtype=torch.bool).cuda()
        pos_mask = targets.expand(
            targets.shape[0], n
        ).t() == targets.expand(n, targets.shape[0])
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_

        groups = []
        for i in range(n):
            it = torch.tensor([i], device=ps.device)
            pos_pair_idx = torch.nonzero(pos_mask[i, i:]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                combinations = get_combinations(it, pos_pair_idx + i)
                groups.append(combinations)

        groups = torch.cat(groups, dim=0)
        p1, p2 = ps[groups[:, 0]], ps[groups[:, 1]]
        z1, z2 = zs[groups[:, 0]], zs[groups[:, 1]]

        loss = (self.criterion(p1, z2) + self.criterion(p2, z1)) * 0.5
        return loss
