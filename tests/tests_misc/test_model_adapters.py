import unittest
from qkeras.simple_qkeras_models import create_simple_qkeras_model
from torch.simple_torch_models import SimpleTorchModel
from src.misc.model_adapters import TorchAdapter, QKerasAdapter
import os


class TestModelAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_model = TorchAdapter(SimpleTorchModel())
        cls.qkeras_model = QKerasAdapter(create_simple_qkeras_model())
        cls.test_file_name = "tmp_test_onnx_file.onnx"

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.test_file_name)

    def test_param_count(self):
        with self.subTest("Trainable"):
            torch_trainable_params = self.torch_model.count_parameters(trainable=True)
            qkeras_trainable_params = self.qkeras_model.count_parameters(trainable=True)
            self.assertEqual(torch_trainable_params, qkeras_trainable_params)

        with self.subTest("Non-Trainable"):
            torch_non_trainable_params = self.torch_model.count_parameters(
                trainable=False
            )
            qkeras_non_trainable_params = self.qkeras_model.count_parameters(
                trainable=False
            )
            self.assertEqual(torch_non_trainable_params, qkeras_non_trainable_params)

    def test_save_to_onnx(self):
        with self.subTest("Torch to ONNX"):
            self.torch_model.save_onnx(file_path="tmp_test_onnx_file.onnx")
        with self.subTest("QKeras to ONNX"):
            self.qkeras_model.save_onnx(file_path="tmp_test_onnx_file.onnx")


if __name__ == "__main__":
    unittest.main()

    # # TODO: write a test for this

    print("dummy")
#
