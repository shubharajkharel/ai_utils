from keras.layers import Input
from keras.models import Model
import unittest
import os
from keras.models import Sequential
from keras.layers import Dense
from src.qkeras.keras_sample_io import KerasSampleIO


class TestKerasSampleIO(unittest.TestCase):
    def test_rand_input_tuple(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(20,)))
        io_instance = KerasSampleIO(model)
        rand_input = io_instance.rand_input()
        self.assertEqual(rand_input.shape, (20,))

    def test_rand_input_list(self):
        # Create a dummy Keras model with multiple inputs
        input1 = Input(shape=(20,))
        input2 = Input(shape=(30,))
        x1 = Dense(10)(input1)
        x2 = Dense(10)(input2)
        model = Model(inputs=[input1, input2], outputs=[x1, x2])

        io_instance = KerasSampleIO(model)
        rand_input = io_instance.rand_input()

        # Ensure that the rand_input is a list and matches the expected input shapes
        self.assertIsInstance(rand_input, list)
        self.assertEqual(rand_input[0].shape, (20,))
        self.assertEqual(rand_input[1].shape, (30,))

    def test_save_io_data_tuple(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(20,)))
        io_instance = KerasSampleIO(model)

        dir_path = "test_sample_io_data"
        input_data_path = os.path.join(dir_path, "model_input.npy")
        output_data_path = os.path.join(dir_path, "model_output.npy")
        try:
            # temp save dir
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            # Save the input and output data
            io_instance.save_io_data(input_data_path, output_data_path)
            self.assertTrue(os.path.exists(input_data_path))
            self.assertTrue(os.path.exists(output_data_path))
        finally:
            if os.path.exists(input_data_path):
                os.remove(input_data_path)
            if os.path.exists(output_data_path):
                os.remove(output_data_path)
            try:
                if os.path.exists(dir_path):
                    os.rmdir(dir_path)
            except Exception as e:
                raise e


if __name__ == "__main__":
    unittest.main()
