"""
A small CNN that predicts a 6-DoF pose (3 rotation + 3 translation values)
directly from a single DRR/X-ray image -- the learned counterpart to the
classical gradient-descent registration in pose_estimation.py.

Given a single forward pass, this predicts the pose in constant time,
instead of iterating for dozens of steps -- the property that makes a
learned approach suitable for real-time use.
"""
import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 200 -> 100
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 100 -> 50
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 50 -> 25
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),  # -> 1x1 (global average pool)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 6),  # 3 rotation + 3 translation, normalized to [-1, 1]
        )

    def forward(self, x):
        return self.regressor(self.features(x))