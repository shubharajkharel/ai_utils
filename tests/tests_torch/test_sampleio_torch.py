import unittest
import os
from torch.simple_torch_models import SimpleTorchCNNModel
from src.torch.torch_sample_io import PyTorchSampleIO


class TestPyTorchSampleIO(unittest.TestCase):
    def test_pytorch_sample_io(self):
        model = SimpleTorchCNNModel()
        sample_io = PyTorchSampleIO(model)
        self.assertIsNotNone(sample_io.sample_input)
        self.assertIsNotNone(sample_io.sample_output)

    def test_batched_input(self):
        model = SimpleTorchCNNModel()
        sample_io = PyTorchSampleIO(model)
        batch_input = sample_io.batched_input(sample_io.sample_input, batch_size=2)
        self.assertEqual(batch_input.shape[0], 2)

    def test_rand_input(self):
        model = SimpleTorchCNNModel()
        sample_io = PyTorchSampleIO(model)
        random_input = sample_io.rand_input()
        self.assertIsNotNone(random_input)

    def test_save_io_data(self):
        model = SimpleTorchCNNModel()
        sample_io = PyTorchSampleIO(model)
        in_file, out_file = sample_io.save_io_data(
            "temp_torch_input.pt", "temp_torch_output.pt"
        )
        self.assertTrue(os.path.exists(in_file))
        self.assertTrue(os.path.exists(out_file))
        os.remove(in_file)
        os.remove(out_file)


if __name__ == "__main__":
    unittest.main()
