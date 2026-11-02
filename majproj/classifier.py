"""Small PointNet that predicts the object class (bottle / box) from the partial scan alone.
Trained only on simulated scans (scripts/train_classifier.py)."""
import torch
import torch.nn as nn

N_POINTS = 512


class PointNetCls(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):  # x: B N 3 (normalized partial) -> logits B C
        return self.head(self.feat(x.transpose(1, 2)).max(-1)[0])
