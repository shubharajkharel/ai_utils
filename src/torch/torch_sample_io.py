from src.misc.sample_io import SampleIO


import torch


class PyTorchSampleIO(SampleIO):
    def rand_input(self):
        if hasattr(self.model, "input_shape"):
            input_shape = self.model.input_shape
        else:
            raise NotImplementedError(
                f"pytorch module requires input_shape attribute \
                    because pytorch does not have a built-in way to \
                        get the input shape of a model"
            )
        return torch.rand(*input_shape)

    def save_data(self, data, file_path):
        torch.save(data, file_path)

    def predict(self, input):
        return self.model(input)


if __name__ == "__main__":
    from src.misc.simple_models import SimpleTorchCNNModel

    model_torch = SimpleTorchCNNModel()
    in_sample, out_sample = PyTorchSampleIO(model_torch)
    print(in_sample)
    print(out_sample)
