import unittest
from src.nn_config import QuantizationConfig, Neuron, NeuronLayer


class TestNeuronLayer(unittest.TestCase):
    def test_layer_neuron_quantization_mismatch_warning(self):
        layer_quant_config = QuantizationConfig(bits=8)
        neuron_quant_config = QuantizationConfig(bits=16)

        neuron1 = Neuron(quantization=neuron_quant_config)
        neuron2 = Neuron()

        with self.assertWarns(UserWarning):
            NeuronLayer([neuron1, neuron2], layer_quantization=layer_quant_config)


if __name__ == "__main__":
    unittest.main()
