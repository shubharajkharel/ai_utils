from keras.layers import Concatenate, Flatten, Input, MaxPooling2D
from keras.models import Model
from qkeras import QConv2D, QDense
from .quantization import QuantizationConfig


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
