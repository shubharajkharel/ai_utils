import unittest
import torch
from xas.nn_config import (
    FeedForwardNNConfig,
    NeuronLayer,
    Neuron,
)
from xas.torch_dynamic_fc import TorchDynamicFC


class TestTorchConfigurableFC(unittest.TestCase):
    def test_forward_pass(self):
        neuron1 = Neuron()
        neuron2 = Neuron()
        neuron3 = Neuron()
        layer1 = NeuronLayer([neuron1, neuron2])
        layer2 = NeuronLayer([neuron2, neuron3])

        config = FeedForwardNNConfig([layer1, layer2])
        model = TorchDynamicFC(config, output_size=3)

        # Create a random tensor of shape (batch_size, input_features)
        x = torch.rand((5, 2))
        output = model(x)

        self.assertEqual(output.shape, (5, 3))

    def test_output_layer_size(self):
        neuron1 = Neuron()
        neuron2 = Neuron()
        layer1 = NeuronLayer([neuron1, neuron2])

        config = FeedForwardNNConfig([layer1])
        model = TorchDynamicFC(config, output_size=4)

        x = torch.rand((5, 2))
        output = model(x)

        self.assertEqual(output.shape, (5, 4))


if __name__ == "__main__":
    unittest.main()
