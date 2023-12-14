import torch
import torch.nn.functional as F


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
