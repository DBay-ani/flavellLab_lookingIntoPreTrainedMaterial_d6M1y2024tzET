#!/bin/sh
HYDRA_FULL_ERROR=1 time python3  src/runInferenceOnPatches_d5M2y2025.py 'data.batch_size=1' 'default_device="cpu"' 'model.args.batch_size=1' 'model.args.cpu=True' 'model.args.n_GPUs=0' 'trainer.accelerator="cpu"' '~trainer.devices'

