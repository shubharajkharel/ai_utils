import logging
from typing import Any, Optional

# import lightning as pl # this giving error about lib typing_extensions
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.model_adapters import TorchAdapter


# TODO: seperate logging and unit testing
class PLModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.02,
        example_input_array: Optional[Any] = None,
        loss_function: Optional[nn.Module] = nn.CrossEntropyLoss(),
        optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam,
        save_graph: bool = False,
    ):
        super().__init__()
        self.model = model
        # learning rate must be set for tuner to find best lr
        self.learning_rate = learning_rate
        self.example_input_array = example_input_array
        self.loss_function = loss_function
        self.optimizer = optimizer
        if hasattr(self.model, "example_input"):  # I had been using this attribute
            self.example_input_array = self.model.example_input
        self.save_graph = save_graph

    def backward(self, loss):
        # added to fix error: trying to backward through the graph a second time
        loss.backward(retain_graph=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # when forward is implemented user has to do eval() and no_grad() ???
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        # log histogram of weights
        if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
            weight_list = TorchAdapter(self).get_weights().values()
            weights = [weight.flatten() for weight in weight_list]
            self.logger.experiment.add_histogram(
                tag="weights",
                values=torch.cat(weights),
                global_step=self.current_epoch,
            )
        else:
            logging.getLogger().warn(
                "Logging weight histogram for non-Tensorboard Logger not implemented"
            )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def on_test_start(self):
        # accumulate predictions and labels for pr curve log
        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self.model(data)
        self.test_preds.append(preds)
        self.test_targets.append(target)

    def on_test_end(self):
        if self.save_graph:
            # log model graph if example input is defined
            if hasattr(self, "example_input_array"):
                self.logger.experiment.add_graph(self, self.example_input_array)
            else:
                log_message = "Logging model graph requires an example input"
                log_message += " ('example_input_array' or 'example_input')"
                log_message += " defined in model. Skipping logging model graph."
                logging.getLogger().info(log_message)
        # log pr curve
        preds = torch.cat(self.test_preds).squeeze()
        targets = torch.cat(self.test_targets).squeeze()
        self.logger.experiment.add_pr_curve(
            tag="pr_curve",
            labels=targets,
            predictions=preds,
        )


if __name__ == "__main__":
    from .pl_data_module import PlDataModule
    from ..torch.sample_torch_dataset import SampleTorchDataset
    from ..torch.simple_torch_models import SimpleTorchCNNModel

    nn_module = SimpleTorchCNNModel()
    dataset = SampleTorchDataset(nn_module)
    data_module = PlDataModule(dataset)
    pl_module = PLModule(SimpleTorchCNNModel())

    data_module.setup()
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(pl_module, datamodule=data_module)
    trainer.test(datamodule=data_module)

    # tuner to find best batch size and inital learning rate
    # from lightning import Tuner
    # tuner = Tuner(trainer)
    # find best batch size by scaling batch size until OOM (out-of-memory)
    # tuner.scale_batch_size(pl_module, datamodule=data_module)
    # find best initial learning rate
    # data_module.setup()
    # lr_finder = tuner.lr_find(
    #     pl_module, train_dataloaders=data_module.train_dataloader()
    # )
    # fig = lr_finder.plot(suggest=True).show()
    # inital_lr = lr_finder.suggestion()
    # pl_module.learning_rate = inital_lr

    # # for OSX only
    # import os
    # import warnings
    # current_value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    # if current_value:
    #     if current_value != "1":
    #         warnings.warn(
    #             f"Environment variable PYTORCH_ENABLE_MPS_FALLBACK is set \
    #               to {current_value}, not '1'"
    #         )
    # else:
    #     os.system('echo "export PYTORCH_ENABLE_MPS_FALLBACK=1"')
    # cpu training needed for Mac M series
    # trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
