from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.data.components.dataset_loadFromNumAssignmentList import UNiFluorInferDataset; 


class UNiFluorInferDataModule(LightningDataModule):
    """`LightningDataModule` for UNiFluorInferDataModule 

    TODO: description 

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        pathToAssignmentCSV: str ,
        trainNumIDs: List[int],
        valNumIDs: List[int],
        testNumIDs: List[int],
        numberOfPatchesPerImage : int, \
        patchSize: int, \
        batch_size: int = 64,
        device: str="",
        num_workers: int = 0,
        pin_memory: bool = False, 
        inch: int = 50
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.trainNumIDs=trainNumIDs;
        self.valNumIDs=valNumIDs;
        self.testNumIDs=testNumIDs;

        self.pathToAssignmentCSV=pathToAssignmentCSV;

        self.batch_size_per_device = batch_size

        self.default_device=device;

    @property
    def num_classes(self) -> int:
        """Get the number of classes.
        """
        raise NotImplementedError();
        return

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        #raise NotImplementedError();
        #MNIST(self.hparams.data_dir, train=True, download=True)
        #MNIST(self.hparams.data_dir, train=False, download=True)
        return;

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """


        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if(self.trainer.world_size > 1):
                raise NotImplementedError("We currently do not support multimachine training or data preperation for UNiFluorInfer");
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            ### default_device=torch.zeros(0).device; # hack for dealing with Pytorch version 2.1 that
            ###     # we are stuck with due to ANTSUN dependencies. In Pytorch version 2.5, there is the
            ###     # function torch.get_default_device() .
            for thisKey, assignedIDNumsToLoad in [ ("data_train", self.trainNumIDs),  
                                           ("data_val", self.valNumIDs),
                                           ("data_test", self.testNumIDs) ]:    
                self.__dict__[thisKey] = UNiFluorInferDataset(pathToAssignmentCSV=self.pathToAssignmentCSV,\
                                                              assignedIDNumsToLoad=assignedIDNumsToLoad, \
                                                              numberOfPatchesPerImage=self.hparams.numberOfPatchesPerImage, \
                                                              patchSize=self.hparams.patchSize, \
                                                              inch=self.hparams.inch);
    
            return;

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # See https://discuss.pytorch.org/t/runtimeerror-expected-a-cuda-device-type-for-generator-but-found-cpu/161463 for 
        # fix below with generator. TODO: save that site/link with InternetArchive
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            generator=torch.Generator(device=self.default_device)
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            generator=torch.Generator(device=self.default_device)
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            generator=torch.Generator(device=self.default_device)
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = UNiFluorInferDataModule()
