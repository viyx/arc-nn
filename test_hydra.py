import hydra
from omegaconf import DictConfig
import torch_xla.distributed.xla_multiprocessing as xmp
from mingpt.model import GPT
# from argparse import ArgumentParser
from datasets import MaxNDataset, ColorPermutation


def add_train_args(parent_parser):
    # TODO check add_positions 
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=10) # log
    parser.add_argument("--val_steps", type=int, default=15000) # make validation every
    parser.add_argument("--log_dir", type=str) # tensorboard and checkpoint dir
    parser.add_argument("--fast_run", action='store_true') # use fast dataset with no transformations
    parser.add_argument("--log_console", action='store_true') # enable logging
    parser.add_argument("--scale_lr", action='store_true') # mult lr by num_cores as sm.optimizer sums batches grads(see https://github.com/pytorch/xla/issues/1781#issuecomment-601849130)
    parser.add_argument("--debug", action="store_true") # work with power on TPU, set device to cpu
    parser.add_argument("--grad_norm_clip", type=float, default=1.0)
    parser.add_argument("--save", action='store_true')
    return parser
# parser = add_train_args(ArgumentParser())
parser = None
# parser = MaxNDataset.add_data_specific_args(parser)
# parser = GPT.add_model_specific_args(parser)
# FLAGS, unknown = parser.parse_known_args()

# m = GPT(FLAGS)
# MODEL = xmp.MpModelWrapper(m)
# SERIAL_EXEC = xmp.MpSerialExecutor()


def get_dataset(fast_run):
        if(fast_run):
            train_trans = None
            test_trans = None
            val_trans = None
        else:
            train_trans = [ColorPermutation(max_colors=FLAGS.n_colors, limit=1000)]
            test_trans = None
            val_trans = [ColorPermutation(max_colors=FLAGS.n_colors, limit=100)]
        ds = MaxNDataset(transforms=[train_trans, test_trans, val_trans], config=FLAGS)
        train, test, val = ds.datasets
        logger.info("Dataset lenghts: train = {}, test = {}, val = {}".format(len(train), len(test), len(val)))
        return train, test, val

def execute():
    global SERIAL_EXEC, MODEL
    # train, test, val = SERIAL_EXEC.run(lambda: get_dataset(FLAGS.fast_run))


@hydra.main(config_name="config")
def my_app(cfg: DictConfig) -> None:
    xmp.spawn(execute, start_method='spawn', nprocs=2)

if __name__ == "__main__":
    my_app()
