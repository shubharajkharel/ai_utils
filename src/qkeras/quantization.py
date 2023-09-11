from math import ceil, floor
from dataclasses import dataclass
from qkeras.quantizers import quantized_bits
from typing import List, Tuple, Union


@dataclass
class QuantizationConfig:
    weight_bits: int = 8
    bias_bits: int = 8
    activation_bits: int = 8

    @property
    def weight_quantizer(self):
        return quantized_bits(self.weight_bits)

    @property
    def bias_quantizer(self):
        return quantized_bits(self.bias_bits)

    @property
    def activation_quantizer(self):
        return quantized_bits(self.activation_bits, 0, 1)
