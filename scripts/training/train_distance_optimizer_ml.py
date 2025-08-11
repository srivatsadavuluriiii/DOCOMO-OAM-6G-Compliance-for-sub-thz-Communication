#!/usr/bin/env python3
"""
Simple trainer for MLDistanceOptimizer using logged tuples.

This script expects a CSV/JSONL of tuples with columns/fields matching
the selected feature list and a target mode. It trains a small MLP and
exports a TorchScript model to disk for inference in MLDistanceOptimizer.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TupleDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], feature_list: List[str], min_mode: int, max_mode: int):
        self.x = []
        self.y = []
        self.feature_list = feature_list
        self.min_mode = min_mode
        self.max_mode = max_mode
        for s in samples:
            feats = [float(s.get(k, 0.0)) for k in feature_list]
            mode = int(s.get('target_mode', s.get('mode', min_mode)))
            mode = max(min_mode, min(max_mode, mode))
            self.x.append(feats)
            self.y.append(mode - min_mode)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int] = [64, 64]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to JSONL with training tuples')
    parser.add_argument('--output', type=str, required=True, help='Output path for TorchScript model (.pt)')
    parser.add_argument('--features', type=str, default='distance,throughput,current_mode,sinr_db', help='Comma-separated feature names')
    parser.add_argument('--min-mode', type=int, default=1)
    parser.add_argument('--max-mode', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    feature_list = [s.strip() for s in args.features.split(',') if s.strip()]
    samples = load_jsonl(args.data)
    if len(samples) == 0:
        raise ValueError('No samples loaded')

    dataset = TupleDataset(samples, feature_list, args.min_mode, args.max_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SimpleMLP(in_dim=len(feature_list), out_dim=(args.max_mode - args.min_mode + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
        avg = total_loss / len(dataset)
        print(f'Epoch {epoch+1}/{args.epochs} - loss {avg:.4f}')

    # Export TorchScript
    model.eval()
    example = torch.randn(1, len(feature_list))
    ts = torch.jit.trace(model, example)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ts.save(args.output)
    print(f'Saved TorchScript model to {args.output}')


if __name__ == '__main__':
    main()


