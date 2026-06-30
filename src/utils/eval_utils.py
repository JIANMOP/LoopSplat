# ── Must be first: force non-interactive backend ──────────────────────
import os as _os
_os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

# Patch evo's SETTINGS.plot_backend BEFORE evo.tools.plot is imported.
# evo/tools/ has no __init__.py, so importing settings directly is safe.
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = 'Agg'

import json
import os
import evo
import numpy as np

import torch
from errno import EEXIST
from os import makedirs, path
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from tqdm import tqdm
