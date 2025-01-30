import torch
from torch import nn


class My_UNiFMIR_variant(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        output_size: int = 10,
    ) -> None:

        super().__init__()



        AAAA = torch.load(cfg.args.pre_train.upDim)
        BBBB = torch.load(cfg.args.pre_train.projection)

        unimodel = model.UniModel(cfg.args); 

        unimodel.load_state_dict(AAAA, strict=False);
        unimodel.load_state_dict(BBBB, strict=False);



        self.model = unimodel;
        # for index in [0,1]:
        #    self.model[index].bias.data = 0 * self.model[index].bias.data;
        #    self.model[index].bias.requires_grad = False; 
        #    self.model[index].weight.data = 0 * self.model[index].weight.data;
            
        # self.model[0].weight.requires_grad = False; 
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, numChannels_obs, xSize, ySize, zSize = x.size()

        x = x.view(batch_size, -1)

        yInitial= self.model(x)

        yFinal=yInitial.view(batch_size, 1, xSize, ySize, zSize)

        return yFinal ;

if __name__ == "__main__":
    _ = SimpleDenseNet()
