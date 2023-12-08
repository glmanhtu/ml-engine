import torch


class LossCombination(torch.nn.Module):
    def __init__(self, criterions):
        super().__init__()
        self.criterions = criterions

    def forward(self, embeddings, targets):
        losses = []
        for criterion in self.criterions:
            losses.append(criterion(embeddings, targets))

        return sum(losses)


class NegativeCosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CosineSimilarity(dim=1)

    def forward(self, predict, actual):
        return -self.criterion(predict, actual).mean()
