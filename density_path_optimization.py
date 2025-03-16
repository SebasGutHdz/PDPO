


from pathlib import Path
project_root = Path(__file__).parent.parent.absolute()
import sys
sys.path.append(str(project_root))

import os
import yaml
import torch
import wandb
import torch.nn.functional as F

from ema_pytorch import EMA
from datetime import datetime
import seaborn as sns

from parametric_pushforward.obstacles import obstacle_cost_stunnel, obstacle_cost_vneck, obstacle_cost_gmm,congestion_cost,geodesic,quadartic_well
from parametric_pushforward.opinion import PolarizeDyn
from parametric_pushforward.spline import Assemble_spline
from parametric_pushforward.visualization import path_visualization_snapshots,disimilarity_snapshots

