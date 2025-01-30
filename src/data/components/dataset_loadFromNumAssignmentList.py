import random
import imageio
import glob
import torch.utils.data as data
from PIL import Image
import torch
import sys
import os
import numpy as np
from tifffile import imread, imsave
from scipy.ndimage.interpolation import zoom
# sys.path.append('../../..')
from csbdeep.utils import normalize, axes_dict, axes_check_and_normalize, backend_channels_last, move_channel_for_backend
import typing;
from src.utils.contracts import requires, ensures;
import re;
from hydra.utils import get_original_cwd ;

import nrrd; 

class UNiFluorInferDataset(data.Dataset):

    @staticmethod
    def _checkFilePathsLoaded(dirPaths : typing.List[str], subFilePaths : typing.Tuple[str,...]) -> None:
        requires(isinstance(dirPaths,list));
        requires(all([isinstance(x,str) for x in dirPaths]));
        requires(isinstance(subFilePaths, tuple));
        requires(all([isinstance(x, str) for x in subFilePaths]));
        if(len(dirPaths) == 0):
            raise Exception("The list of directory paths is empty.");
        for thisDirPath in dirPaths:
            if(not os.path.exists(thisDirPath)):
                raise Exception(f"Path \"{thisDirPath}\" does not exist.");
            if(not os.path.isdir(thisDirPath)):
                raise Exception(f"Path \"{thisDirPath}\" exists but is not a directory.");
            if(not thisDirPath.endswith("/")):
                raise Exception("We expect the directory path to end in a \"/\".")
            for thisSubFile in subFilePaths:
                if(re.match("^[a-zA-Z0-9_\.]+$", thisSubFile) is None):
                    raise Exception(\
                        f"Proposed file name \"{thisSubFile}\" does not match name pattern"+\
                         " allowed, \"^[a-zA-Z0-9_\.]+$\" .");
                completePathToFile=thisDirPath+thisSubFile;
                if(not os.path.exists(completePathToFile)):
                    raise Exception(f"Path \"{completePathToFile}\" does not exist.");
                if(not os.path.isfile(completePathToFile)):
                    raise Exception(f"Path \"{completePathToFile}\" exists but is not a file.");
        return;


    def __init__(self,\
        pathToAssignmentCSV :str,\
        assignedIDNumsToLoad : typing.List[int],\
        dtype=torch.float64,neuropal_ch_to_grab_indx=1):
        print("\n\n\n" + pathToAssignmentCSV + "\n\n\n", flush=True)
        requires(isinstance(pathToAssignmentCSV, str));
        requires(len(pathToAssignmentCSV)> 0);
        requires(os.path.exists(pathToAssignmentCSV));
        requires(os.path.isfile(pathToAssignmentCSV));
        self.confocalVolumeDims=[290, 115, 72];
        self.numChannels_obs=1;
        self.numChannels_label=1;
        self.neuropal_ch_to_grab_indx=neuropal_ch_to_grab_indx;
        self.numChInitialNeuropal=3;
        self.assignedIDNumsToLoad=set(assignedIDNumsToLoad);
        self.dtype=dtype;
        self.priorLoaded=dict();
        self.pathToSplitSpecification=pathToAssignmentCSV;
        self.usesOfSubFilesAndTheirPaths : typing.Tuple[typing.Tuple[str,str],typing.Tuple[str,str]] = \
                                                 [("obs", "all_red.nrrd"), ("target", "NeuroPAL.nrrd")];
        fh=open(self.pathToSplitSpecification,"r");
        dirPaths=[];
        for thisLine in fh.read().split("\n"):
            if(thisLine == ""):
                continue;
            numIDOfSplit, pathToDirectory = thisLine.split(",");
            if(int(numIDOfSplit) not in assignedIDNumsToLoad):
                continue;
            dirPaths.append(get_original_cwd() + "/data/"+pathToDirectory);
        self._checkFilePathsLoaded(dirPaths,tuple([x[1] for x in self.usesOfSubFilesAndTheirPaths]));
        self._numberInstances=len(dirPaths);
        self._dirPaths=dirPaths;
        return;

    def __len__(self) -> int:
        return self._numberInstances;

    def __getitem__(self, idx):
        if(idx < 0):
            raise Exception("This dataloader does not support negative indexing");
        if(idx>=len(self._dirPaths)):
            raise IndexError(f"Specified index {idx} is outside the bounds of list of dirPaths list of length {self._numberInstances}.");
        # obs[:,:,:,:] = idx;
        # target[:,:,:,:] = idx+1;
        if(idx in self.priorLoaded):
            # BELOW LINE ASSUMES THAT THE CALLER WILL NOT MUTATE THE VALUES 
            # PROVIDED IN readNRRDs["obs"] AND readNRRDs["target"]
            return self.priorLoaded[idx];
        dirName=self._dirPaths[idx];
        readNRRDs=dict();
        # TODO: check the dimensions more carefully...
        for thisVar, thisFileName in self.usesOfSubFilesAndTheirPaths:
            temp = nrrd.read(dirName + thisFileName, index_order="F"); #, dtype=self.dtype);
            readNRRDs[thisVar] = temp[0];
            if(thisVar=="target"):
                # assert(temp[0].shape == tuple(self.confocalVolumeDims + [3]));
                readNRRDs[thisVar] = readNRRDs[thisVar][:,:,:,self.neuropal_ch_to_grab_indx];
            # readNRRDs[thisVar]= torch.from_numpy(readNRRDs[thisVar]).to(dtype=self.dtype).reshape(*([1] + self.confocalVolumeDims))
            temp123=torch.from_numpy(readNRRDs[thisVar]).to(dtype=self.dtype)
            print(f"\n\n{thisFileName}:{temp123.shape}")
            readNRRDs[thisVar]=torch.zeros(tuple([1] + self.confocalVolumeDims),dtype=self.dtype); #temp123.reshape(*([1] + list(temp123.shape)))
            readNRRDs[thisVar][0,:(temp123.shape[0]),:(temp123.shape[1]),:(temp123.shape[2])] = temp123
            # assert(readNRRDs[thisVar].shape == tuple([1] + self.confocalVolumeDims));
            assert(isinstance(readNRRDs[thisVar] , torch.Tensor));
            assert(readNRRDs[thisVar].dtype == self.dtype );
            assert(readNRRDs[thisVar].requires_grad == False );
        
        # BELOW LINE ASSUMES THAT THE CALLER WILL NOT MUTATE THE VALUES 
        # PROVIDED IN readNRRDs["obs"] AND readNRRDs["target"]
        self.priorLoaded[idx]=(readNRRDs["obs"], readNRRDs["target"]);
        
        return readNRRDs["obs"], readNRRDs["target"] ; #, filename
    

    
