from src.misc.sample_io import SampleIO


import numpy as np


class KerasSampleIO(SampleIO):
    def rand_input(self):
        input_shape = self.model.input_shape
        if isinstance(input_shape, tuple):
            return np.random.rand(*input_shape[1:])  # Exclude batch size
        elif isinstance(input_shape, list):
            return [
                np.random.rand(*inp[1:]) for inp in input_shape
            ]  # Exclude batch size
        elif isinstance(input_shape, dict):
            return {
                key: np.random.rand(*shape[1:]) for key, shape in input_shape.items()
            }  # Exclude batch size
        else:
            raise NotImplementedError(
                f"Input shape of type {type(input_shape)} not implemented"
            )

    def save_data(self, data, file_path):
        np.save(file_path, data)

    def predict(self, input):
        return self.model.predict(input)


if __name__ == "__main__":
    from qkeras.simple_qkeras_models import create_simple_qkeras_model

    model_keras = create_simple_qkeras_model()
    in_sample, out_sample = KerasSampleIO(model_keras)
    print("Keras")
    print(in_sample)
    print(out_sample)
