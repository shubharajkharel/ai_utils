from typing import List

from qkeras import QDense, quantized_bits
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Concatenate, Input, Reshape
from tensorflow.keras.models import Sequential

from utils.src.optuna.nn_config import (FeedForwardNNConfig, Neuron,
                                        NeuronLayer, QuantizationConfig)


# Function to generate QDense layer with quantization
def create_qdense_layer(units, input_shape=None, quantization_config=None):
    quantizer = None
    if quantization_config:
        quantizer = quantized_bits(
            quantization_config.weight_bits, quantization_config.bias_bits
        )
    if input_shape:
        return QDense(
            units,
            input_shape=input_shape,
            kernel_quantizer=quantizer,
            bias_quantizer=quantizer,
        )
    else:
        return QDense(units, kernel_quantizer=quantizer, bias_quantizer=quantizer)


# Function to create fully connected network
def create_fully_connected_qkeras_model(
    widths: List[int],
    layer_quantizations: List[int],
    output_size: int = 1,
    output_quantization: int = 8,
):
    model = Sequential()
    input_shape = [
        widths[0],
    ]

    for i, (width, layer_quantization) in enumerate(zip(widths, layer_quantizations)):
        quantization_config = QuantizationConfig(bits=layer_quantization)
        if i == 0:
            model.add(
                create_qdense_layer(
                    width,
                    input_shape=input_shape,
                    quantization_config=quantization_config,
                )
            )
        else:
            model.add(
                create_qdense_layer(width, quantization_config=quantization_config)
            )

        if i < len(widths) - 1:
            model.add(Activation("relu"))

    output_quantization_config = QuantizationConfig(bits=output_quantization)
    model.add(
        create_qdense_layer(output_size, quantization_config=output_quantization_config)
    )
    return model


if __name__ == "__main__":
    widths = [16, 3, 3]
    layer_quantizations = [8, 16, 32]
    # layer_quantizations = [8] * len(widths)
    model = create_fully_connected_qkeras_model(
        widths=widths,
        layer_quantizations=layer_quantizations,
        output_size=1,
        output_quantization=8,
    )
    print(model.summary())
    
    
    from pprint import pprint
    for layer in model.layers:
        if isinstance(layer, QDense):  # Or whatever quantized layer you're interested in
            pprint(f"Layer {layer.name}:")
            pprint(f"  Kernel Quantizer: {layer.kernel_quantizer}")
            pprint(f"  Bias Quantizer: {layer.bias_quantizer}")

