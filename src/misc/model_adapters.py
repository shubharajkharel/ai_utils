from ..qkeras.keras_sample_io import KerasSampleIO
import onnx
from ..qkeras.simple_qkeras_models import create_simple_qkeras_model
import torchinfo
import torch
import numpy as np
from abc import ABC, abstractmethod
from tf2onnx.convert import from_keras

# import keras2onnx  # Warning: Repo archived and not working
# https://www.codeproject.com/Articles/5278501/Making-Keras-Models-Portable-Using-ONNX


class BaseAdapter(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def predict(self, data, *args, **kwargs):
        pass

    @abstractmethod
    def count_parameters(self, trainable=True):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def save_onnx(self):
        pass

    @abstractmethod
    def sample_io(self):
        pass

    @abstractmethod
    def get_weights(self):
        pass


class TorchAdapter(BaseAdapter):
    def predict(self, data, channel_last=True):
        if channel_last and isinstance(data, np.ndarray):
            data = torch.tensor(
                data.transpose(0, 3, 1, 2)
            )  # Convert channel_last to channel_first
        return self.model(data)

    def count_parameters(self, trainable=True):
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad == trainable
        )

    def summary(self, *args, **kwargs):
        input_data = (
            self.model.example_input if hasattr(self.model, "example_input") else None
        )
        return torchinfo.summary(
            self.model, *args, verbose=0, input_data=input_data, **kwargs
        )

    def save_onnx(self, file_path="torch_model.onnx", example_input=None, **kwargs):
        if example_input is None:
            if hasattr(self.model, "example_input"):
                example_input = self.model.example_input
            else:
                raise ValueError(
                    "Example Input must be provided while saving to onnx, or the model \
                        must have an example_input attribute"
                )
        torch.onnx.export(model=self.model, args=example_input, f=file_path, **kwargs)

    def sample_io(self):
        raise NotImplementedError("Sample IO not implemented for Torch")

    def get_weights(self):
        weights_dict = {}
        for name, param in self.model.named_parameters():
            weights_dict[name] = param.data
        return weights_dict


class QKerasAdapter(BaseAdapter):
    def predict(self, data, channel_first=True):
        if channel_first and isinstance(data, torch.Tensor):
            data = (
                data.detach().numpy().transpose(0, 2, 3, 1)
            )  # Convert channel_first to channel_last
        return self.model.predict(data)

    def count_parameters(self, trainable=True):
        if trainable:
            return sum([np.prod(p.shape) for p in self.model.trainable_weights])
        else:
            return sum([np.prod(p.shape) for p in self.model.layers if not p.trainable])

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def save_onnx(self, file_path="keras_model.onnx", example_input=None, **kwargs):
        # https://onnxruntime.ai/docs/tutorials/tf-get-started.html
        onnx_model, _ = from_keras(model=self.model)
        onnx.save(onnx_model, f=file_path)

    def sample_io(self, sample_input=None):
        return iter(KerasSampleIO(self.model, sample_input))

    def get_weights(self):
        raise NotImplementedError("Weights not implemented for QKeras")


if __name__ == "__main__":
    from ..torch.simple_torch_models import SimpleTorchModel

    torch_model = TorchAdapter(SimpleTorchModel())
    qkeras_model = create_simple_qkeras_model()

    print(torch_model.summary())
    print(qkeras_model.summary())
