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

from csbdeep.data.generate import create_patches


from torch.nn.functional import interpolate ;

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
        numberOfPatchesPerImage : int, \
        patchSize : int,
        dtype=torch.float32,neuropal_ch_to_grab_indx=1,\
        inch=50):
        print("\n\n\n" + pathToAssignmentCSV + "\n\n\n", flush=True)
        requires(isinstance(pathToAssignmentCSV, str));
        requires(len(pathToAssignmentCSV)> 0);
        requires(os.path.exists(pathToAssignmentCSV));
        requires(os.path.isfile(pathToAssignmentCSV));
        # For the below
        # (UniFMIR_d15M1y2025) david@flv-c3:/storage/fs/store1/david/lookingIntoPreTrainedMaterial_d6M1y2024tzET$ time ( head /data1/prj_starvation/data_processed/*/NRRD/*_t0308_ch1.nrrd | grep sizes | sort | uniq )
        # sizes: 322 212 77
        # (UniFMIR_d15M1y2025) david@flv-c3:/storage/fs/store1/david/lookingIntoPreTrainedMaterial_d6M1y2024tzET$ date
        # Thu Jan 30 17:59:37 UTC 2025
        # (UniFMIR_d15M1y2025) david@flv-c3:/storage/fs/store1/david/lookingIntoPreTrainedMaterial_d6M1y2024tzET$ 
        self.confocalVolumeDims=[322, 216, 80]# [360, 120, 80];
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
        self._numberInstances=len(dirPaths)* numberOfPatchesPerImage;
        self._dirPaths=dirPaths;

        self.patchSize=(inch,patchSize,patchSize); # patchSize);
        self.numberOfPatchesPerImage=numberOfPatchesPerImage;
        return;

    def __len__(self) -> int:
        return self._numberInstances;

    def __getitem__(self, idx):
        if(idx < 0):
            raise Exception("This dataloader does not support negative indexing");
        if(idx>=self._numberInstances): # len(self._dirPaths)):
            raise IndexError(f"Specified index {idx} is outside the number of instances we have of length {self._numberInstances}"); #  bounds of list of dirPaths list of length {self._numberInstances}.");
        # obs[:,:,:,:] = idx;
        # target[:,:,:,:] = idx+1;
        if(idx in self.priorLoaded):
            # BELOW LINE ASSUMES THAT THE CALLER WILL NOT MUTATE THE VALUES 
            # PROVIDED IN readNRRDs["obs"] AND readNRRDs["target"]
            return self.priorLoaded[idx];
        dirName=self._dirPaths[idx // self.numberOfPatchesPerImage];
        readNRRDs=dict();
        # TODO: check the dimensions more carefully...
        for thisVar, thisFileName in self.usesOfSubFilesAndTheirPaths:
            temp = nrrd.read(dirName + thisFileName, index_order="F"); #, dtype=self.dtype);
            readNRRDs[thisVar] = temp[0];
            if(thisVar=="target"):
                # assert(temp[0].shape == tuple(self.confocalVolumeDims + [3]));
                readNRRDs[thisVar] = readNRRDs[thisVar][:,:,:,self.neuropal_ch_to_grab_indx];
            presentSize=readNRRDs[thisVar].shape;
            assert(len(presentSize) == 3);
            longestAxisTarget=64 *3;
            shortAxis= max(64, int((presentSize[1]/presentSize[0])* longestAxisTarget));
            rightHandSideToReturn = interpolate(\
                (torch.Tensor(readNRRDs[thisVar])).view((1,1, readNRRDs[thisVar].shape[0], readNRRDs[thisVar].shape[1], readNRRDs[thisVar].shape[2])), \
                (longestAxisTarget ,shortAxis, 70), \
                mode="trilinear");
            # To make use of the fact that the patch-sizes of the pre-trained networks we return are
            # (50, 64, 64).
            assert(rightHandSideToReturn.shape[0:2] == (1,1));
            rightHandSideToReturn=rightHandSideToReturn[0,0,:,:,:];
            rightHandSideToReturn =torch.transpose(rightHandSideToReturn, dim0=0,dim1=2)
            readNRRDs[thisVar] = rightHandSideToReturn; 

        numberOfPatchesPerImage=self.numberOfPatchesPerImage

        class exampleRawData():

            def generator(self):
                def gen():
                    yield readNRRDs["obs"], readNRRDs["target"], "XYZ", None ; #readNRRDs["obs"].to("cpu").numpy(), readNRRDs["target"].to("cpu").numpy(), "XYZ", None;

                return gen();

            @property
            def size(self):
                return 1; ### the patch generation code expects that the number here reflects the number of raw images, not the number of patches....  # numberOfPatchesPerImage;
    
            @property
            def description(self):
                return "Internal class for forming patches of the data."


        patches = create_patches( exampleRawData(), self.patchSize, self.numberOfPatchesPerImage, patch_filter=None, shuffle=False);
        assert(len(patches) == 5);

        # BELOW LINE ASSUMES THAT THE CALLER WILL NOT MUTATE THE VALUES 
        # PROVIDED IN readNRRDs["obs"] AND readNRRDs["target"]
        for subInd in range(0,self.numberOfPatchesPerImage):
            newSubInd=idx - (idx % self.numberOfPatchesPerImage)+ subInd;
            thisObs=torch.Tensor(patches[0][subInd, :,:,:]).view(*self.patchSize).numpy(); #.to("cuda:0");
            thisTarget=torch.Tensor(patches[1][subInd, :,:,:]).view(*self.patchSize).numpy(); # .to("cuda:0");
            rest_thisObs=torch.Tensor(patches[3][subInd, :,:,:]).view(*readNRRDs["obs"].shape).numpy();
            rest_thisTarget=torch.Tensor(patches[4][subInd, :,:,:]).view(*readNRRDs["target"].shape).numpy();
            self.priorLoaded[newSubInd]=(thisObs, thisTarget, rest_thisObs, rest_thisTarget); # readNRRDs["obs"], readNRRDs["target"]);
        
        return self.priorLoaded[idx]; #readNRRDs["obs"], readNRRDs["target"] ; #, filename
    

    
