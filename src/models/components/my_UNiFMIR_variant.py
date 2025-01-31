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

        for x in unimodel.parameters():
            # print(str(x.flatten()[0]))
            x.data = torch.nan * x.data ;

        for thisPath in []: #weights_to_load_in_order:
            print("thisPath:" + str(thisPath),flush=True);
            theseWeights = torch.load(thisPath);
            unimodel.load_state_dict(theseWeights, strict=False);
        
        for x in unimodel.parameters():
            if(torch.any(torch.isnan(x.data))):
                x.requires_grad=False;

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
