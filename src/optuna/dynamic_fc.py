import torch.nn as nn
from .nn_config import (
    FeedForwardNNConfig,
    NeuronLayer,
    Neuron,
)
from utils.src.lightning.pl_module import PLModule
from typing import List


class TorchDynamicFC(nn.Module):
    def __init__(self, widths: List[int], output_size: int = 1):
        fc_config = FeedForwardNNConfig(
            [NeuronLayer([Neuron()] * width) for width in widths]
        )
        super().__init__()
        # super(TorchDynamicFC, self).__init__()
        self.layers = nn.ModuleList()
        for i, layer in enumerate(fc_config.neuron_layers):
            layer_in = len(layer.neurons)
            try:
                layer_out = len(fc_config.neuron_layers[i + 1].neurons)
            except IndexError:  # Last layer
                layer_out = output_size
            self.layers.append(nn.Linear(layer_in, layer_out))
            if i < len(fc_config.neuron_layers) - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PlDynamicFC(PLModule):
    def __init__(self, widths: List[int], output_size: int = 1, **kwargs):
        model = TorchDynamicFC(widths, output_size=output_size)
        self.widths = widths
        self.output_size = output_size
        # model.input_shape = (widths[0],)
        # model.example_input = PyTorchSampleIO(model=model) # TODO device error
        super().__init__(model=model, **kwargs)


if __name__ == "__main__":
    # Create FeedForwardNNConfig
    neuron1 = Neuron(quantization=1)  # should be ignored
    neuron2 = Neuron()

    layer1 = NeuronLayer([neuron1, neuron2])
    layer2 = NeuronLayer([neuron2] * 10)

    config = FeedForwardNNConfig([layer1, layer2])

    # Create NN model from config
    model = TorchDynamicFC(config, output_size=2)
    print(model)
