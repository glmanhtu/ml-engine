import torch


class DistanceLoss(torch.nn.Module):
    def __init__(self, loss_fn, distance_fn=None):
        super().__init__()
        self.criterion = loss_fn
        if distance_fn is None:
            self.distance_fn = loss_fn
        else:
            self.distance_fn = distance_fn

    def forward(self, predict, actual):
        return self.criterion(predict, actual)

    def compute_distance(self, predict, actual):
        return self.distance_fn(predict, actual)


class BatchDotProduct(torch.nn.Module):
    # See https://github.com/pytorch/pytorch/issues/18027

    def forward(self, predict, actual):
        B, S = predict.shape
        return torch.bmm(predict.view(B, 1, S), actual.view(B, S, 1)).reshape(-1)


class LossCombination(torch.nn.Module):
    def __init__(self, criterions):
        super().__init__()
        self.criterions = criterions

    def forward(self, embeddings, targets):
        losses = []
        for criterion in self.criterions:
            losses.append(criterion(embeddings, targets))

        return sum(losses)


class NegativeLoss(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, predict, actual):
        return -self.criterion(predict, actual).mean()


class NegativeCosineSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CosineSimilarity(dim=1)

    def forward(self, predict, actual):
        return -self.criterion(predict, actual).mean()
