import torch.nn as nn
from src.nn_config import (
    FeedForwardNNConfig,
    NeuronLayer,
    Neuron,
)


class TorchDynamicFC(nn.Module):
    def __init__(self, config: FeedForwardNNConfig, output_size: int = 1):
        super(TorchDynamicFC, self).__init__()
        self.layers = nn.ModuleList()
        for i, layer in enumerate(config.neuron_layers):
            layer_in = len(layer.neurons)
            try:
                layer_out = len(config.neuron_layers[i + 1].neurons)
            except IndexError:  # Last layer
                layer_out = output_size
            self.layers.append(nn.Linear(layer_in, layer_out))
            if i < len(config.neuron_layers) - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
