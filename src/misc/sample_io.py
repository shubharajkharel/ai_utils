import torch
import numpy as np
from abc import ABC, abstractmethod


class SampleIO(ABC):
    def __init__(self, model, sample_input=None,device = None):
        self.model = model
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_input = self.rand_input() if not sample_input else sample_input
        self.sample_input = self.sample_input.to(device)
        self.sample_output = self.predict(self.batched_input(self.sample_input))
        self.sample_output = self.sample_output.to(device)

    def batched_input(self, inp, batch_size=1):
        if isinstance(inp, np.ndarray):
            return np.expand_dims(inp, axis=0)
        elif isinstance(inp, list):
            return [self.batched_input(i, batch_size) for i in inp]
        elif isinstance(inp, dict):
            return {key: self.batched_input(i, batch_size) for key, i in inp.items()}
        elif isinstance(inp, torch.Tensor):
            return torch.cat([torch.unsqueeze(inp, 0)] * batch_size, dim=0)
        else:
            raise NotImplementedError(
                f"Input type {type(inp)} not implemented for batching"
            )

    def save_io_data(
        self,
        model_input_file="model_input.npy",
        model_output_file="model_output.npy",
    ):
        self.save_data(self.sample_input, model_input_file)
        self.save_data(self.sample_output, model_output_file)
        return model_input_file, model_output_file

    def __iter__(self):
        return iter((self.sample_input, self.sample_output))

    @abstractmethod
    def save_data(self, data, file_name, file_format="npy"):
        pass

    @abstractmethod
    def predict(self, input):
        pass

    @abstractmethod
    def rand_input(self):
        pass
