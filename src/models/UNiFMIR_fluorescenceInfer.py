from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from torchmetrics.image import StructuralSimilarityIndexMeasure;

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
from torch.utils.data import dataloader
import model

import os
from decimal import Decimal
import torch.nn.utils as utils
# import imageio
from utility import savecolorim
import numpy as np
from src.data.components.UNiFluorInferDataset import normalize, PercentileNormalizer
# from tifffile import imsave

import logging
import hydra

from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning import Trainer
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


rp = os.path.dirname(__file__)



class UNiFluorInferModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        args
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.nanCounts={x : 0 for x in ["x", "y", "logits", "loss"] };

        # loss function
        ### self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        ## self.train_acc = Accuracy(task="multiclass", num_classes=10)
        ## self.val_acc = Accuracy(task="multiclass", num_classes=10)
        ## self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()

        self.ssim= StructuralSimilarityIndexMeasure();

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        xPassForward=x;
        if(torch.any(torch.isnan(x))):
            xPassForward=torch.mean(x[~torch.isnan(x)])*torch.ones(*x.shape);
            xPassForward[~torch.isnan(x)] = x[~torch.isnan(x)];
            xPassForward.require_grad=False;
        xPassForward=xPassForward + torch.rand(*xPassForward.shape) * 0.02 * torch.mean(xPassForward);
        logits=torch.ones(1) * torch.nan;
        for iterationNum in range(0,20):
            logits = self.forward(xPassForward);
            if(not torch.any(torch.isnan(logits))):
                break;
            xPassForward=0.9* xPassForward;
            y=0.9*y;
            self.optimizers().zero_grad()



        ### utility.compute_psnr_and_ssim(logits, y);
        
        ####print("(logits.shape, y.shape):" + str((logits.shape, y.shape)), flush=True);

        """
        absDiff= torch.abs(logits- (y-x)); #(y/(x-+1.0))**2);
        loss = torch.sum(absDiff[~torch.isnan(absDiff)]);#torch.max(logits ** 2); # torch.max((logits-y) ** 2) # replaced a torch.sum that was here with a torch.max to see if that addressed the issue with nans appearing# self.criterion(logits, y)
        """
        assert(logits.shape == (x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]));
        logits=logits[:,0,:,:];
        loss = -self.ssim(logits.reshape(y.shape), y.reshape(y.shape)); #.reshape(y.shape))
        #### logits2= torch.max(torch.zeros(*y.shape), logits - torch.quantile(logits,0.7)) * 100;   #(logits/(torch.max(logits)+0.01)) ** 4;
        #### y2=torch.max(torch.zeros(*y.shape), y - torch.quantile(y,0.7)) * 100    #(y/(torch.max(y)+0.01))**4;
        ### loss =loss-self.ssim(logits2.reshape(y.shape), y2.reshape(y.shape));
        ### brightnessMask=(y <= torch.quantile(y,0.60));
        ### brightnessMask=( logits > y) & brightnessMask;
        ### extraBrightnessPenalty=torch.sum(torch.abs(logits[brightnessMask] - y[brightnessMask]) );
        ### loss = loss * (x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3])  + extraBrightnessPenalty
        ### preds = torch.argmax(logits, dim=1)
        for val in ["x", "y", "logits", "loss"]:
            if(torch.any(torch.isnan(eval(val)))):
                print("torch.isnan("+val+") is True", flush=True);
                self.nanCounts[val] = 1 + self.nanCounts.get(val,0) ;
        return loss, logits, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        ## self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        ## self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/param/finalRes", self.net.model.coeffForFinalAddition_x2d.data, on_step=False, on_epoch=True, prog_bar=True); 
        for val in self.nanCounts.keys():
            self.log("train/nanCounts/"+val, self.nanCounts[val], on_step=False, on_epoch=True, prog_bar=True);
        # return loss or backpropagation will fail
        LRs=[x["lr"] for x in self.optimizers().state_dict()["param_groups"]];
        self.log("train/LRs/min", min(LRs), on_step=False, on_epoch=True, prog_bar=True);
        self.log("train/LRs/max", max(LRs), on_step=False, on_epoch=True, prog_bar=True);
        for q in range(1,10):
            self.log("train/LRs/q0."+str(q), np.quantile(LRs, (q/10)), on_step=False, on_epoch=True, prog_bar=True);
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        ## self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        ## self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        for val in self.nanCounts.keys():
            self.log("val/nanCounts/"+val, self.nanCounts[val], on_step=False, on_epoch=True, prog_bar=True);



    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass;
        # ## acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # elf.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        #self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        for val in self.nanCounts.keys():
            self.log("test/nanCounts/"+val, self.nanCounts[val], on_step=False, on_epoch=True, prog_bar=True);
        
        # print(str(self.optimizers().state_dict()), flush=True); # .state_dict()), flush=True);


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizationVals=[];
        # Below line uses reversed since my recollection is that the default order returned
        # by the parameters() function is earliest-registered to latest-registered...
        rateDecrease = (252/ 255);#( (1.0 /  8 ) * 6 ); # 1/8 is representable fully in float, and the values as chosen here get the
                                            # gradients to be about 5% of their value after 10 layers etc.
        rateDecrease=rateDecrease**2;
        for index, param in enumerate(reversed([ x for x in self.trainer.model.parameters()])):
            thisLR = self.hparams.args.lr * ( rateDecrease ** (index // 2)); ### //2 to account for the typical weights+bias combination
            optimizationVals.append({"params": param, "lr": thisLR});
        optimizer = self.hparams.optimizer( # params=self.trainer.model.parameters())
                optimizationVals                 )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


