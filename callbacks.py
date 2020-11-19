
import math
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mingpt.utils import sample_while, plot_task

class SampleImagesCallback(pl.Callback):
    def on_test_end(self, trainer, model):
        # trainer = self
        # model = trainer.get_model()
        # trainer.accelerator_backend.barrier('sample')
        # if(trainer.global_rank == 0):
            # print()
        dl = model.test_dataloader()
        print(dl.sampler)
        # gpt_dataset = pl_module.test_dataloader().dataset
        # dl = DataLoader(gpt_dataset, batch_size=16*16, num_workers=16, drop_last=False, shuffle=False)
        results = sample_while(model, dl)
        tasks = dl.dataset.dataset.tasks

        assert len(tasks) == len(results)

        # for i in range(tasks):
        # fig = plot_task(gpt_dataset.dataset, )
        # import torch
        # im = torch.randint(0, gpt_dataset.vocab_size, (10,10))
        tensorboard = model.logger.experiment
        for i, task in enumerate(tasks):
            res = results[i]
            fig = plot_task(dl.dataset.dataset, task, res)
            tensorboard.add_figure(task, fig, trainer.global_step, True)