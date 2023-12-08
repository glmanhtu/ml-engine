import torch
import torch.nn as nn
from torchvision import models


class SimSiam(nn.Module):
    """
    Adapted from https://github.com/facebookresearch/simsiam
    Build a SimSiam model.
    """
    def __init__(self, arch, pretrained, dim=2048, pred_dim=512, dropout=0):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        base_encoder = models.__dict__[arch]

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        if pretrained is None:
            self.encoder = base_encoder(num_classes=dim, zero_init_residual=True, weights=pretrained)
        else:
            self.encoder = base_encoder(weights=pretrained)
            num_ftrs = self.encoder.fc.in_features  # Getting last layer's output features
            self.encoder.fc = nn.Linear(num_ftrs, dim)

        # Modify the average pooling layer to use a smaller kernel size
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Dropout(p=dropout),
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[7].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x):
        """
        Input:
            x: Stacked of two images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        x1, x2 = torch.unbind(x, 1)

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()


class SimSiamV2(SimSiam):
    def forward(self, x):
        z1 = self.encoder(x) # NxC
        p1 = self.predictor(z1) # NxC
        return p1, z1.detach()


class SimSiamV2CE(nn.Module):

    def __init__(self, arch, pretrained, n_classes, dim=2048, pred_dim=512, dropout=0):
        super().__init__()
        base_encoder = models.__dict__[arch]

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        if pretrained is None:
            self.encoder = base_encoder(num_classes=dim, zero_init_residual=True, weights=pretrained)
        else:
            self.encoder = base_encoder(weights=pretrained)
            num_ftrs = self.encoder.fc.in_features  # Getting last layer's output features
            self.encoder.fc = nn.Linear(num_ftrs, dim)

        # Modify the average pooling layer to use a smaller kernel size
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = self.encoder.fc
        self.encoder.fc = nn.Identity()

        # build a 3-layer projector
        prev_dim = self.fc.weight.shape[1]
        self.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Dropout(p=dropout),
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.fc[7].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.classifier = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(prev_dim, prev_dim // 2, bias=False),
                                        nn.BatchNorm1d(prev_dim // 2),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(prev_dim // 2, n_classes))

    def forward(self, x):
        x = self.encoder(x)
        z1 = self.fc(x)
        p1 = self.predictor(z1)
        cls = self.classifier(x)
        return p1, z1.detach(), cls
