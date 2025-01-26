from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


import faulthandler

faulthandler.enable()

import typing

import lightning
import hydra

import torch
import utility
import loss

torch.backends.cudnn.enabled = True
import argparse
from mydata import FlouresceneVCD, Flouresceneproj, MyExampleDataLoader
from torch.utils.data import dataloader
import model

import os
from decimal import Decimal
import torch.nn.utils as utils
# import imageio
from utility import savecolorim
import numpy as np
from mydata import normalize, PercentileNormalizer
# from tifffile import imsave

import logging
import hydra

from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning import Trainer
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


rp = os.path.dirname(__file__)
