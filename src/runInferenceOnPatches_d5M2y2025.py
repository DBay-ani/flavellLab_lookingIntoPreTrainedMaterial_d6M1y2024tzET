

import torch
from torch import nn
from typing import List;
from models.components.UniFMIR.Unimodel import UniModel;
import hydra;


from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# pathToModel="logs/train/runs/2025-02-05_06-12-03/checkpoints/epoch_epoch=590.ckpt"
pathToModel='/home/david/tempForFasterCheckpointing/train/runs/2025-02-06_21-49-41/checkpoints/epoch_epoch=024.ckpt';   

import matplotlib.pyplot as plt
import numpy as np
"""
# Create a sample matrix
matrix = np.random.rand(100, 100)

# Display the matrix as an image
plt.imshow(matrix, cmap='gray')  # You can change the colormap if needed
plt.axis('off')  # Hide the axes

# Save the image
plt.savefig('matrix.png')
"""

@hydra.main(version_base="1.3", config_path="../configs", config_name="runInferenceOnPatches_d5M2y2025.yaml")
def main(cfg: DictConfig): # -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """


    with torch.no_grad():
        # set the default device across all Pytorch tensors
        torch.set_default_device(
            torch.device(cfg.default_device)
        )
    
    
        # set seed for random number generators in pytorch, numpy and python.random
        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)
    
    
    
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
        datamodule.setup();
    
        model: LightningModule = hydra.utils.instantiate(cfg.model)
    
        checkpoint  = torch.load(pathToModel) ### Running the code it said it failed to load for weights-only.... #  , weights_only=True);
    
        print(str(type(checkpoint)) + "\n" + str(checkpoint)[:1000], flush=True);
    
        print("\n\n\n" + str(checkpoint.keys()), flush=True);
    
        model.load_state_dict(checkpoint['state_dict'])
    
        # 4. Set the model to evaluation mode if needed
        model.eval()
    
        flattenFunct=(lambda z: torch.max(z,dim=1)[0].reshape(64,64) );
    
        for indx, batch in enumerate(datamodule.val_dataloader()):
            x,y = batch;
            yHat = model.forward(x);
            print(str([z.shape for z in [x,y,yHat]]),flush=True);
            x=flattenFunct(x);
            y=flattenFunct(y);
            yHat=flattenFunct(yHat[0,:,:,:,:]);
            print(str([z.dtype for z in [x,y,yHat]]),flush=True);
            print(str([z.shape for z in [x,y,yHat]]),flush=True);
            """
            for name in ["x", "y", "yHat"]:
                plt.imshow(eval(name), cmap='gray')  # You can change the colormap if needed
                plt.axis('off')  # Hide the axes
    
                # Save the image
                plt.savefig(str(indx)+"_" + name+'.png')
            """
            
            """for name in ["x", "y", "yHat"]:
                temp=eval(name);
                tail=0.015
                exec( name + " = (temp-torch.quantile(temp,tail))/(torch.quantile(temp,1-tail) - torch.quantile(temp,tail))");
                exec( name + " = torch.clamp(" + str(name) + ", 0, 1)");
                print(str([ f(eval(name)) for f in [torch.min, torch.max, torch.median, torch.mean, torch.sum]]), flush=True);
            """
            for name in [x, y, yHat]:
                tail=0.015
                name.data = (name.data-torch.quantile(name.data,tail))/(torch.quantile(name.data,1-tail) - torch.quantile(name.data,tail));
                name.data  = torch.clamp( name.data , 0, 1);
                print(str([ f(name) for f in [torch.min, torch.max, torch.median, torch.mean, torch.sum]]), flush=True);


            for val in [x, y,yHat]:
                print(str([ f(val) for f in [torch.min, torch.max, torch.median, torch.mean, torch.sum]]), flush=True);
           
            print("\n");
            together=torch.cat((x,y,yHat),dim=1);
            plt.imshow(together, cmap='gray')
            plt.axis('off');
            plt.savefig("./data/inferenceExamples/652df99d-8110-4ddd-a475-7872344fdab6/" + str(indx)+'_together.png')
            
    return cfg;
    
    
A = main();
