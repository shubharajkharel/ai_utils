from src.qkeras.quantization import QuantizationConfig
from keras.models import Model
from keras.layers import Input, Flatten, Concatenate, ZeroPadding2D
import torch
from torchinfo import summary
from pprint import pprint
from qkeras import QDense, QActivation
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten
from qkeras import QConv2D, QDense, QActivation
import torch.nn as nn

import torch
import torch.nn.functional as F
import torch.nn as nn


# Define a simple fully connected neural network
class SimpleTorchFCModel(nn.Module):
    def __init__(self, input_shape=(10,), output_size=2, hidden_size=30):
        super(SimpleTorchFCModel, self).__init__()

        self.input_size = input_shape
        self.output_size = output_size
        self.hidden_size = hidden_size

        #! Sequential class doesnt seems to work with hls4ml
        self.a = nn.Linear(input_shape, hidden_size)
        self.b = nn.ReLU()
        self.c = nn.Linear(hidden_size, hidden_size)
        self.d = nn.ReLU()
        self.e = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        x = self.e(x)
        return x


class SimpleTorchCNNModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), output_size=10):
        super(SimpleTorchCNNModel, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(320, output_size)

    def forward(self, x):
        x_in = x
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class SimpleTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),  # Include Flatten layer
            nn.Linear(16 * 13 * 13, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.example_input = torch.randn(1, 1, 28, 28)

    def forward(self, x):
        return self.model(x)

    # inputs = [Input(shape=mc.input_shape) for mc in config.mode_configs]


def create_simple_qkeras_model():
    keras_input = Input(shape=(28, 28, 1))
    qconfig = QuantizationConfig()
    x = keras_input
    x = QConv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_quantizer=qconfig.weight_quantizer,
        bias_quantizer=qconfig.bias_quantizer,
    )(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = QDense(
        64,
        activation="relu",
        kernel_quantizer=qconfig.weight_quantizer,
        bias_quantizer=qconfig.bias_quantizer,
    )(x)
    x = QDense(
        10,
        kernel_quantizer=qconfig.weight_quantizer,
        bias_quantizer=qconfig.bias_quantizer,
    )(x)
    return Model(inputs=keras_input, outputs=x)


def create_simple_qkeras_model_with_concat(method="two_once"):
    if method == "two_once":
        input_shapes = [(16, 16, 1), (16, 16, 1)]
        model_input = [Input(shape=shape) for shape in input_shapes]
        x = model_input  # TODO: x should not be needed here
        post_concat_x = Concatenate(axis=-1)(x)
        model = Model(inputs=model_input, outputs=post_concat_x)
    elif method == "one_at_a_time":
        input_shapes = [(16, 16, 1), (16, 16, 1), (16, 16, 1)]
        model_input = [Input(shape=shape) for shape in input_shapes]
        concat_value = model_input[0]
        for i in range(1, len(model_input)):
            concat_value = Concatenate(axis=-1)([concat_value, model_input[i]])
        model = Model(inputs=model_input, outputs=concat_value)
    else:
        raise ValueError(f"Unknown method {method}")
    return model


if __name__ == "__main__":
    model = create_simple_qkeras_model_with_concat(method="one_at_a_time")
    print("Simple QKeras model with iterative binary concat summary:")
    print(model.summary())

    model = create_simple_qkeras_model_with_concat(method="two_once")
    print("Simple QKeras model with single binary concat summary:")
    print(model.summary())

    # model_pytorch = SimpleTorchCompositeModel()
    # print("PyTorch model summary:")
    # summary(model_pytorch, input_size=(1, 1, 28, 28))

    # model = SimpleTorchCNNModel()
    # print(model)
    # print(model(torch.randn(1, 1, 28, 28)))
