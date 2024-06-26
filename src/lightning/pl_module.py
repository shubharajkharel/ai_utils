from typing import Any, Optional

import lightning as pl  # this giving error about lib typing_extensions

# import pytorch_lightning as pl
import torch
import torch.nn as nn


# TODO: seperate logging and unit testing
class PLModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: Optional[float] = 0.0001,
        example_input_array: Optional[Any] = None,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        save_graph: bool = False,
    ):
        super().__init__()
        self.model = model
        # learning rate must be set for tuner to find best lr
        self.learning_rate = learning_rate
        self.example_input_array = example_input_array
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.loss = loss if loss is not None else nn.MSELoss()
        if hasattr(self.model, "example_input"):  # I had been using this attribute
            self.example_input_array = self.model.example_input
        self.save_graph = save_graph
        # self.save_hyperparameters()

    # def backward(self, loss):
    #     # added to fix error: trying to backward through the graph a second time
    #     loss.backward(retain_graph=True)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # when forward is implemented user has to do eval() and no_grad() ???
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self.model(data)
        loss = self.loss(preds, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True)


if __name__ == "__main__":
    from ..torch.sample_torch_dataset import SampleTorchDataset
    from ..torch.simple_torch_models import SimpleTorchCNNModel
    from .pl_data_module import PlDataModule

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
