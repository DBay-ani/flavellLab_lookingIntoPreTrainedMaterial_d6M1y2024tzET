(UniFMIR_d15M1y2025) david@flv-c3:~/store1/david/lookingIntoPreTrainedMaterial_d6M1y2024tzET$ PROJECT_ROOT=$(pwd) python3
Python 3.10.14 | packaged by conda-forge | (main, Mar 20 2024, 12:45:18) [GCC 12.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from hydra import compose, initialize
>>> from omegaconf import OmegaConf
>>> import hydra
>>> with initialize(version_base="1.3", config_path="./configs"):
...     cfg = compose(config_name="train");
... 
>>> A = hydra.utils.instantiate(cfg.data)
>>> A.
A.CHECKPOINT_HYPER_PARAMS_KEY                         A.load_from_checkpoint(                               A.setup(
A.CHECKPOINT_HYPER_PARAMS_NAME                        A.load_state_dict(                                    A.state_dict()
A.CHECKPOINT_HYPER_PARAMS_TYPE                        A.name                                                A.teardown(
A.allow_zero_length_dataloader_with_multiple_devices  A.num_classes                                         A.test_dataloader()
A.batch_size_per_device                               A.on_after_batch_transfer(                            A.train_dataloader()
A.data_test                                           A.on_before_batch_transfer(                           A.trainer
A.data_train                                          A.on_exception(                                       A.transfer_batch_to_device(
A.data_val                                            A.predict_dataloader()                                A.transforms(
A.from_datasets(                                      A.prepare_data()                                      A.val_dataloader()
A.hparams                                             A.prepare_data_per_node                               
A.hparams_initial                                     A.save_hyperparameters(                               
>>> A.data_test
>>> A.data_train
>>> A.setup()
>>> A.data_train
<torch.utils.data.dataset.Subset object at 0x7ff275d92980>
>>> for x in A.data_train:
...     print(str(x)[:100],flush=True);
... 
(array([[[[22., 22., 22., ..., 22., 22., 22.],
         [22., 22., 22., ..., 22., 22., 22.],
       
(array([[[[32., 32., 32., ..., 32., 32., 32.],
         [32., 32., 32., ..., 32., 32., 32.],
       
(array([[[[33., 33., 33., ..., 33., 33., 33.],
         [33., 33., 33., ..., 33., 33., 33.],
       
(array([[[[17., 17., 17., ..., 17., 17., 17.],
         [17., 17., 17., ..., 17., 17., 17.],
       
(array([[[[13., 13., 13., ..., 13., 13., 13.],
         [13., 13., 13., ..., 13., 13., 13.],
       
(array([[[[10., 10., 10., ..., 10., 10., 10.],
         [10., 10., 10., ..., 10., 10., 10.],
       
(array([[[[7., 7., 7., ..., 7., 7., 7.],
         [7., 7., 7., ..., 7., 7., 7.],
         [7., 7., 7
(array([[[[19., 19., 19., ..., 19., 19., 19.],
         [19., 19., 19., ..., 19., 19., 19.],
       
(array([[[[21., 21., 21., ..., 21., 21., 21.],
         [21., 21., 21., ..., 21., 21., 21.],
       
(array([[[[16., 16., 16., ..., 16., 16., 16.],
         [16., 16., 16., ..., 16., 16., 16.],
       
(array([[[[4., 4., 4., ..., 4., 4., 4.],
         [4., 4., 4., ..., 4., 4., 4.],
         [4., 4., 4
(array([[[[2., 2., 2., ..., 2., 2., 2.],
         [2., 2., 2., ..., 2., 2., 2.],
         [2., 2., 2
(array([[[[0., 0., 0., ..., 0., 0., 0.],
         [0., 0., 0., ..., 0., 0., 0.],
         [0., 0., 0
(array([[[[3., 3., 3., ..., 3., 3., 3.],
         [3., 3., 3., ..., 3., 3., 3.],
         [3., 3., 3
(array([[[[29., 29., 29., ..., 29., 29., 29.],
         [29., 29., 29., ..., 29., 29., 29.],
       
(array([[[[27., 27., 27., ..., 27., 27., 27.],
         [27., 27., 27., ..., 27., 27., 27.],
       
(array([[[[26., 26., 26., ..., 26., 26., 26.],
         [26., 26., 26., ..., 26., 26., 26.],
       
(array([[[[30., 30., 30., ..., 30., 30., 30.],
         [30., 30., 30., ..., 30., 30., 30.],
       
(array([[[[8., 8., 8., ..., 8., 8., 8.],
         [8., 8., 8., ..., 8., 8., 8.],
         [8., 8., 8
(array([[[[18., 18., 18., ..., 18., 18., 18.],
         [18., 18., 18., ..., 18., 18., 18.],
       
>>> 

