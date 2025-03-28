import torch;

from src.utils.contracts import requires, ensures;

class LowRank3DTo3DTransform(torch.nn.Module):


    @staticmethod
    def _initializeCloseToIdentity(noiseLevel, targetTorchTensor):
        requires(isinstance(noiseLevel, float));
        requires(noiseLevel >= 0);  
        requires(len(set(targetTorchTensor.weight.shape)) == 1);
        targetTorchTensor.bias.data= noiseLevel * targetTorchTensor.bias.data;
        targetTorchTensor.weight.data= noiseLevel * targetTorchTensor.weight.data + \
                                       torch.eye(targetTorchTensor.weight.shape[0]);
        return ;
  
    def __init__(self,targetSize, noiseLevel):
        # Assuming targetSize is (batch_size, embeddingDim, X, Y);
        requires(isinstance(targetSize, tuple));
        requires(len(targetSize) == 4);
        requires(all([isinstance(x,int) for x in targetSize]));
        requires(min(targetSize) > 0)
        requires(isinstance(noiseLevel, float));
        requires(noiseLevel >= 0);        
        super(LowRank3DTo3DTransform, self).__init__()
        batch_size, self.embedding_size, self.X_size, self.Y_size= targetSize;
        self.embeddingMix = torch.nn.Linear(self.embedding_size, self.embedding_size);
        self.spaceMix = torch.nn.Linear(self.X_size * self.Y_size, self.X_size * self.Y_size);
        self._initializeCloseToIdentity(noiseLevel, self.embeddingMix);
        self._initializeCloseToIdentity(noiseLevel, self.spaceMix);
        return ;
        
    def forward(self, M):
        requires(M.shape[1:] == (self.embedding_size, self.X_size, self.Y_size));
        batch_size=M.shape[0];
        M=M.transpose(1,3);
        M=self.embeddingMix(M);
        M=M.transpose(1,3);
        M=self.spaceMix(M.reshape(batch_size, self.embedding_size, -1)).reshape( \
            batch_size, self.embedding_size, self.X_size, self.Y_size);
        return M ;
        
