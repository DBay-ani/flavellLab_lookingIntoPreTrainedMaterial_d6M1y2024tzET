import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        """
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )
        """
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size,dtype=torch.float16),
            nn.Linear(lin1_size, output_size,dtype=torch.float16)
        );
        for index in [0,1]:
            self.model[index].bias.data = 0 * self.model[index].bias.data;
            self.model[index].bias.requires_grad = False; 
            self.model[index].weight.data = 0 * self.model[index].weight.data;
            
        self.model[0].weight.requires_grad = False; 
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, numChannels_obs, xSize, ySize, zSize = x.size()


        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        yInitial= self.model(x)
        for index in [0,1]:
            for val in ["self.model["+str(index)+"].bias", "self.model["+str(index)+"].weight"]:
                print("torch.std("+val+"):" + str(torch.std(eval(val))), flush=True)
                print("torch.max(torch.abs("+val+")):" + str(torch.max(torch.abs(eval(val)))) + "\n\n", flush=True);
        
        yFinal=yInitial.view(batch_size, 3, xSize, ySize, zSize)

        return yFinal ;

if __name__ == "__main__":
    _ = SimpleDenseNet()
