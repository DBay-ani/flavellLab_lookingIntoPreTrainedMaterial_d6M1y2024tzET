import torch
from torch import nn
from typing import List;
from models.components.UniFMIR.Unimodel import UniModel;

from torch.nn.functional import interpolate ;


class My_UNiFMIR_variant(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    @staticmethod
    def _initHelper_loadModelWeights(weights_to_load_in_order, thisModel):
         #for x in unimodel.parameters():
        #    # print(str(x.flatten()[0]))
        #    x.data = torch.nan * x.data ;

        # NOTE: probably the last scalar will still be nan....

        numAssigned=0;
        numSkipped=0;

        setsOfNamesAssignedTo={x : set() for x in weights_to_load_in_order };
        for thisPath in weights_to_load_in_order:
            print("thisPath:" + str(thisPath),flush=True);
            theseWeights = torch.load(thisPath);
            for thisName, x in thisModel.named_parameters():
                # print(str(x.flatten()[0]))
                if(thisName not in theseWeights):
                    continue;
                if(x.shape == theseWeights[thisName].shape):
                    print(f"ASSIGNING -  weight \"{thisName}\" from file \"{thisPath}\"", flush=True);
                    x.data = theseWeights[thisName];
                    # if(numAssigned % 2 ==0 ):
                    #    x.requires_grad = False;
                    numAssigned=numAssigned+1;
                    setsOfNamesAssignedTo[thisPath].add(thisName);
                else:
                    print(f"Skipping weight \"{thisName}\" from file \"{thisPath}\"; our model has this as size ${x.shape} while the loaded content has size ${theseWeights[thisName].shape}.", flush=True);
                    numSkipped=numSkipped+1;
        
        setsOfNamesAssignedByFiles=list(setsOfNamesAssignedTo.values());
        assignedVals=[len(x) for x in setsOfNamesAssignedByFiles];
        intersections=[];
        # for v1, v2 in [[0,1],[1,2],[0,2]]:
        #     intersections.append(len(setsOfNamesAssignedByFiles[v1].intersection(setsOfNamesAssignedByFiles[v2])));
        print(f"ASSIGNED: {numAssigned}, SKIPPED: {numSkipped}, Lengths of values read: {assignedVals}, VALUES DOUBLE-ASSIGNED TO: {intersections}", flush=True);
 
            # unimodel.load_state_dict(theseWeights, strict=False);
        """
        for x in unimodel.parameters():
            # if("_x2d" in x.name):
            #     continue;
            if(torch.any(torch.isnan(x.data))):
                x.requires_grad=False;
        
        for thisName, x in unimodel.named_parameters():
            if(torch.any(torch.isnan(x.data))):
                print("Named parameter unassigned to: "+ thisName, flush=True);
        
        print("UNAMED PARAMETERS: " + str(len([x for x in unimodel.parameters()]) - len([ x for x in unimodel.named_parameters()]))  , flush=True);
        """
        return ;


    def __init__(
        self,
        weights_to_load_in_order: List[str],
        unimodels 
    ) -> None:
        super().__init__()
        self.localModel=unimodels.localModel.to("cuda:0");
        self.globalModel=unimodels.globalModel.to("cuda:0");

        for thisModel in [self.localModel, self.globalModel]:
            self._initHelper_loadModelWeights(weights_to_load_in_order, thisModel);
        
        # The below line may seem silly, but we need to do this
        # so we can convieniantly mix the intermediate values of these models in the
        # "forward" method of the localModel
        self.localModel.globalModel= self.globalModel; 

        return;

    def forward(self, xPatch: torch.Tensor, xComplementPatch: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # batch_size, xSize, ySize, zSize = x.size()

        #x = x.view(batch_size, -1)

        # Below line is just an example used to check that 
        # both copies of the model are able to proceed as expected
        # and that the maximum memory used does not exceed what the 
        # GPU has.
        sXCP= xComplementPatch.shape;
        sXP=xPatch.shape;
        resized_xComplementPatch=interpolate(xComplementPatch.view(sXCP[0],1, sXCP[1], sXCP[2], sXCP[3]), (sXP[1], sXP[2], sXP[3]), mode="trilinear").squeeze();
        yPatch, yFull = self.localModel(xPatch,resized_xComplementPatch );


        return yPatch, yFull ;
