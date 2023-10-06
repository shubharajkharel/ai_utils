from pytorch_lightning import Callback


import warnings


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