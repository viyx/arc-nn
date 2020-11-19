

# import pdb, sys
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/utilities/xla_device_utils.py')
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/trainer.py')
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/accelerators/tpu_accelerator.py')
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/utilities/xla_device_utils.py')
# pdb.set_trace()



# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

from argparse import ArgumentParser
import numpy as np
from datasets import MedianDataModule
from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from mingpt.model import GPT
# from callbacks import SampleImagesCallback


def cli_main():
    pl.seed_everything(333)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GPT.add_model_specific_args(parser)
    parser = MedianDataModule.add_data_specific_args(parser)
    args = parser.parse_args()
    dm = MedianDataModule.from_argparse_args(args)
    model = GPT(**args.__dict__)
    # cl = SampleImagesCallback()
    # chck_cl = ModelCheckpoint(dirpath='~/projects/arc-nn/lightning_logs/version_48/checkpoints')
    trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=True)
    trainer.fit(model, datamodule=dm)
    # pass
    trainer.test(model, datamodule=dm)

    # model = GPT.load_from_checkpoint('./lightning_logs/version_87')



if __name__=='__main__':
    cli_main()

