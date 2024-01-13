import torch
import torch.nn.functional as F

from ml_engine.utils import get_combinations


class BatchWiseTripletDistanceLoss(torch.nn.Module):

    def __init__(self, distance_fn, margin=0.15):
        super().__init__()
        self.margin = margin
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=distance_fn)

    def forward(self, samples, targets):
        n = samples.size(0)
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.bool).cuda()
        pos_mask = targets.expand(
            targets.shape[0], n
        ).t() == targets.expand(n, targets.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_

        pos_groups, neg_groups = [], []
        for i in range(n):
            it = torch.tensor([i], device=samples.device)
            pos_pair_idx = torch.nonzero(pos_mask[i, i:]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_combinations = get_combinations(it, pos_pair_idx + i)

                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                if pos_combinations.shape[0] > 0 and neg_pair_idx.shape[0] > 0:
                    neg_combinations = get_combinations(it, neg_pair_idx)

                    if pos_combinations.shape[0] < neg_combinations.shape[0]:
                        pos_combinations = pos_combinations[torch.randint(high=pos_combinations.shape[0],
                                                                          size=(neg_combinations.shape[0],))]
                    elif neg_combinations.shape[0] < 1:
                        continue
                    elif neg_combinations.shape[0] < pos_combinations.shape[0]:
                        neg_combinations = neg_combinations[torch.randint(high=neg_combinations.shape[0],
                                                                          size=(pos_combinations.shape[0],))]
                    neg_groups.append(neg_combinations)
                    pos_groups.append(pos_combinations)

        pos_groups = torch.cat(pos_groups, dim=0)
        neg_groups = torch.cat(neg_groups, dim=0)

        assert torch.equal(pos_groups[:, 0], neg_groups[:, 0])
        anchor = samples[pos_groups[:, 0]]
        positive = samples[pos_groups[:, 1]]
        negative = samples[neg_groups[:, 1]]

        return self.loss_fn(anchor, positive, negative)


class BatchWiseTripletLoss(torch.nn.Module):
    """
    Adapted from https://github.com/marco-peer/hip23
    Triplet loss with negative similarity as distance function
    """
    def __init__(self, margin=0.1):
        super(BatchWiseTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, emb, target):
        emb = F.normalize(emb, p=2, dim=-1)
        return self.forward_impl(emb, target, emb, target)

    def forward_impl(self, inputs_col, targets_col, inputs_row, targets_row):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.bool).cuda()
        pos_mask = targets_col.expand(
            targets_row.shape[0], n
        ).t() == targets_row.expand(n, targets_row.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] * ~eyes_

        loss = list()
        neg_count = list()
        for i in range(n):
            pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_pair_ = sim_mat[i, pos_pair_idx]
                pos_pair_ = torch.sort(pos_pair_)[0]

                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                neg_pair_ = sim_mat[i, neg_pair_idx]
                neg_pair_ = torch.sort(neg_pair_)[0]

                select_pos_pair_idx = torch.nonzero(
                    pos_pair_ < neg_pair_[-1] + self.margin
                ).view(-1)
                pos_pair = pos_pair_[select_pos_pair_idx]

                select_neg_pair_idx = torch.nonzero(
                    neg_pair_ > max(0.6, pos_pair_[-1]) - self.margin
                ).view(-1)
                neg_pair = neg_pair_[select_neg_pair_idx]

                pos_loss = torch.sum(1 - pos_pair)
                if len(neg_pair) >= 1:
                    neg_loss = torch.sum(neg_pair)
                    neg_count.append(len(neg_pair))
                else:
                    neg_loss = 0
                loss.append(pos_loss + neg_loss)
            else:
                loss.append(0)

        loss = sum(loss) / n
        return loss
