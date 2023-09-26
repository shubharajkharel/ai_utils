from typing import Union
import warnings
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from utils.src.misc.model_adapters import PLAdapter


class TensorboardLogTestTrainLoss(Callback):
    """Logs two scalars in same plot: train_loss and val_loss"""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"]
        pl_module.logger.experiment.add_scalars(
            "losses", {"train_loss": loss}, trainer.global_step
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        pl_module.logger.experiment.add_scalars(
            "losses", {"val_loss": val_loss}, trainer.global_step
        )


class TensorboardLogAllWeigthsHist(Callback):
    """Logs histogram of weight at end of training"""

    def on_train_end(self, trainer, pl_module):
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


class TensorboardLogGraph(Callback):
    """Trainer logger needs to be Tensorboard Logger"""

    def on_test_end(self, trainer, pl_module):
        if hasattr(pl_module, "example_input_array"):
            pl_module.logger.experiment.add_graph(
                pl_module, pl_module.example_input_array
            )
        else:
            warning_msg = "Logging model graph requires an example input"
            warning_msg += " ('example_input_array' or 'example_input')"
            warning_msg += " defined in model. Skipping logging model graph."
            raise warnings.warn(warning_msg)
