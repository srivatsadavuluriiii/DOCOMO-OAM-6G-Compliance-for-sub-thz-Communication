"""
Unified Visualization Module for OAM 6G

This module consolidates all visualization functionality into a single,
consistent interface.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy.special import assoc_laguerre

                                            
from .visualization_consolidated import (
    plot_training_curves,
    create_interactive_dashboard,
    visualize_q_values,
    plot_comparison,
    plot_metrics_grid,
    plot_heatmap
)

                                       
create_evaluation_dashboard = create_interactive_dashboard

                                                           
class MetricsLogger:
    """
    Metrics logger supporting JSONL and optional TensorBoard backends.

    Usage:
      logger = MetricsLogger(log_dir, backends=["jsonl","tensorboard"], flush_interval=1)
      logger.log_scalar("reward", 1.23, step=10)
      logger.log_episode_summary({"episode_reward": 100.0}, episode=1)
      logger.close()
    """

    def __init__(self, log_dir: str, backends: Optional[List[str]] = None, flush_interval: int = 1):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.backends = backends or ["jsonl"]
        self.flush_interval = max(int(flush_interval), 1)
        self._jsonl_path = os.path.join(self.log_dir, "metrics.jsonl")
        self._jsonl_fh = open(self._jsonl_path, "a", encoding="utf-8")
        self._tb_writer = None
        if "tensorboard" in self.backends:
            try:
                self._tb_writer = SummaryWriter(self.log_dir)
            except Exception:
                self._tb_writer = None
        self._buffered = 0

    def log_scalar(self, tag: str, value: float, step: int) -> None:
               
        entry = {"type": "scalar", "tag": tag, "value": float(value), "step": int(step)}
        self._jsonl_fh.write(f"{__import__('json').dumps(entry)}\n")
        self._buffered += 1
        if self._buffered >= self.flush_interval:
            self._jsonl_fh.flush()
            self._buffered = 0
                     
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_scalar(tag, float(value), int(step))
            except Exception:
                pass

    def log_episode_summary(self, summary: Dict[str, Any], episode: int) -> None:
        entry = {"type": "episode", "episode": int(episode), **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in summary.items()}}
        self._jsonl_fh.write(f"{__import__('json').dumps(entry)}\n")
        self._jsonl_fh.flush()
                                             
        if self._tb_writer is not None:
            for k in ("episode_reward", "episode_throughput", "episode_handovers"):
                if k in summary:
                    try:
                        self._tb_writer.add_scalar(k, float(summary[k]), int(episode))
                    except Exception:
                        pass

    def close(self) -> None:
        try:
            if self._tb_writer is not None:
                self._tb_writer.close()
        finally:
            try:
                self._jsonl_fh.flush()
            finally:
                self._jsonl_fh.close()

                      
__all__ = [
                     
    'MetricsLogger',
    
                            
    'plot_training_curves',
    'create_interactive_dashboard',
    'visualize_q_values',
    'plot_comparison',
    'plot_metrics_grid',
    'plot_heatmap',
    
                          
    'create_evaluation_dashboard',
] 
