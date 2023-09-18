from pprint import pprint
import warnings
from dataclasses import dataclass
from typing import Union, List, Optional


@dataclass
class QuantizationConfig:
    weight_bits: Optional[int] = None
    bias_bits: Optional[int] = None
    activation_bits: Optional[int] = None

    def __init__(
        self,
        bits: Union[int, None] = None,
        weight_bits: Optional[int] = None,
        bias_bits: Optional[int] = None,
        activation_bits: Optional[int] = None,
    ):
        if bits is not None:
            self.weight_bits = self.bias_bits = self.activation_bits = bits
        else:
            self.weight_bits = weight_bits
            self.bias_bits = bias_bits
            self.activation_bits = activation_bits


@dataclass
class Neuron:
    quantization: Optional[QuantizationConfig] = None


@dataclass
class NeuronLayer:
    neurons: List[Neuron]
    layer_quantization: Optional[QuantizationConfig] = None

    def __post_init__(self):
        if self.layer_quantization:
            for neuron in self.neurons:
                if (
                    neuron.quantization
                    and neuron.quantization != self.layer_quantization
                ):
                    warnings.warn(
                        "Layer and neuron quantization settings differ. Layer settings will override."
                    )


@dataclass
class FeedForwardNNConfig:
    neuron_layers: List[NeuronLayer]


if __name__ == "__main__":
    neuron_quant_config = QuantizationConfig(bits=16)

    neuron1 = Neuron(quantization=neuron_quant_config)
    neuron2 = Neuron()

    layer_quant_config = QuantizationConfig(bits=8)
    layer = NeuronLayer([neuron1, neuron2], layer_quantization=layer_quant_config)

    config = FeedForwardNNConfig([layer])

    pprint(config)
