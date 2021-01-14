import os
from omegaconf import OmegaConf
import hydra
from wandb.util import generate_id


@hydra.main(config_path='conf', config_name="config")
def main(cfg):
    import torch_xla.distributed.xla_multiprocessing as xmp
    from test_hydra import map_fn
    assert os.environ['WANDB_API_KEY'], 'Specify wandb api key'

    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ['XLA_USE_BF16'] = '1'
    if(cfg.debug):
        os.environ['XRT_DEVICE_MAP'] = 'CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0'
        os.environ['XRT_WORKERS'] = 'localservice:0;grpc://localhost:40934'
    else:
        assert os.environ['XRT_TPU_CONFIG'], 'Specify xla device.'
    
    # allow to change
    # OmegaConf.set_struct(cfg, False)
    full_conf = hydra.core.hydra_config.HydraConfig.get()

    xmp.spawn(map_fn, args=(cfg, full_conf,), nprocs=cfg.n_cores, start_method='spawn')


if __name__ == '__main__':
    # OmegaConf.register_resolver("wandb_id", lambda: generate_id())
    main()