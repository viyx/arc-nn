

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
import pytorch_lightning as pl
from mingpt.model import GPT


def cli_main():
    pl.seed_everything(333)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GPT.add_model_specific_args(parser)
    parser = MedianDataModule.add_specific_args(parser)
    args = parser.parse_args()
    dm = MedianDataModule.from_argparse_args(args)
    model = GPT(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args)
    # trainer = Trainer(tpu_cores=args.num_cores,
    #                     precision=args.precision,
    #                     max_epochs=args.,
    #                     # fast_dev_run=True,
    #                     limit_train_batches=0.02,
    #                     # limit_val_batches=100,
    #                     # val_check_interval=1000,
    #                     # gradient_clip_val=1.0,
    #                     # callbacks=[lr_decay],
    #                     progress_bar_refresh_rate=1)

    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)


if __name__=='__main__':
    # print(123)
    # Trainer(tpu_cores=1)
    cli_main()

