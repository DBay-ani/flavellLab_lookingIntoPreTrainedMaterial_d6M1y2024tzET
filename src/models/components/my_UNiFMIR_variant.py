import torch
from torch import nn
from typing import List;
from models.components.UniFMIR.Unimodel import UniModel;

class My_UNiFMIR_variant(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        weights_to_load_in_order: List[str],
        unimodel : nn.Module
    ) -> None:

        super().__init__()

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
            for thisName, x in unimodel.named_parameters():
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
        self.model = unimodel.to("cuda:0"); 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        # batch_size, xSize, ySize, zSize = x.size()

        #x = x.view(batch_size, -1)

        yInitial= self.model(x)

        # yFinal=yInitial.view(batch_size, 1, xSize, ySize, zSize)
        yFinal=yInitial;

        return yFinal ;
