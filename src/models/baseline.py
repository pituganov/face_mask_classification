"""Скрипт с baseline моделью

Можно и нужно использовать в качестве основы для своего решения
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 2)

    def forward(self, x):
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        return x
