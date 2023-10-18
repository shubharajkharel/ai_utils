# import pytorch_lightning as pl

import lightning as pl
import torch
from pytorch_lightning import Callback


class TensorboardLogAllWeigthsHist(Callback):
    """Logs histogram of weight at end of training"""

    def on_train_end(self, trainer, pl_module):
        if isinstance(
            pl_module.logger, pl.pytorch.loggers.tensorboard.TensorBoardLogger
        ):
            # comment out to avoid unnecessary dependencies
            from utils.src.misc.model_adapters import PLAdapter

            weight_list = PLAdapter(pl_module).get_weights().values()
            weights = [weight.flatten() for weight in weight_list]

            # weights = []
            # for _, param in pl_module.named_parameters():
            #     weights.append(param.flatten())

            pl_module.logger.experiment.add_histogram(
                tag="weights",
                values=torch.cat(weights),
                global_step=pl_module.current_epoch,
            )
        else:
            raise NotImplementedError(
                "Logging weight histogram for non-Tensorboard Logger not implemented"
            )
