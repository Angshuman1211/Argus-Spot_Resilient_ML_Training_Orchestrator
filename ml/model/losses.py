import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Spot interruptions are rare. If we use standard BCE loss, the model will
    just output 0 (no interruption) every time and achieve 99% accuracy.
    Focal loss forces the model to heavily penalize missing the rare disruptions.
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_logit_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_logit_loss(inputs, targets)
        pt = torch.exp(-bce_loss)  # probability of the correct class
        # alpha_t: alpha for positive targets, (1 - alpha) for negative ones. The old
        # code applied self.alpha as a flat scalar to every sample regardless of its
        # label - that's mathematically a no-op for class balance (equivalent to
        # scaling the learning rate), not the "heavily penalize the rare class" effect
        # the docstring above claims. This is the per-class weighting that actually
        # does it.
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
