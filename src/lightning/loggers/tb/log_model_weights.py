# import pytorch_lightning as pl
import lightning as pl
import torch
from pytorch_lightning import Callback


import logging


class TensorboardLogAllWeigthsHist(Callback):
    """Logs histogram of weight at end of training"""

    def on_train_end(self, trainer, pl_module):
        from utils.src.misc.model_adapters import PLAdapter

        if isinstance(pl_module.logger, pl.loggers.tensorboard.TensorBoardLogger):
            weight_list = PLAdapter(pl_module).get_weights().values()
            weights = [weight.flatten() for weight in weight_list]
            pl_module.logger.experiment.add_histogram(
                tag="weights",
                values=torch.cat(weights),
                global_step=pl_module.current_epoch,
            )
        else:
            logging.getLogger().warn(
                "Logging weight histogram for non-Tensorboard Logger not implemented"
            )