from data.generate import create_patches
import numpy as np;


class exampleRawData():

    def __init__(self):
        self.numExamples=3;
    
    def generator(self):
        def gen():
            for x in range(1,self.numExamples + 1):
                # Note that the create_patches function performs normalizations, so in the case that 
                # np.ones was used below instead of np.random.rand, the patches returned would be all zero
                # (I tried it).
                yield (x * np.random.rand(*(1,8,16,32))), (1 - x * np.random.rand(*(1,8,16,32))), "CXYZ", None;

        return gen();

    @property
    def size(self):
        return self.numExamples;

    def description(self):
        "Example data loader to get the patch creation in order."

AAAA = create_patches(exampleRawData(), (1,8,16,16), 4,patch_filter=None)