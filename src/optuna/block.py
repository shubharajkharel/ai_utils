import torch.nn as nn
from torch.nn import ModuleDict
import torch


from typing import List, Tuple
from typing import TypeVar

BlockType = TypeVar("BlockType", bound="Block")


class Block(nn.Module):
    def __init__(
        self,
        name: str = "Block",
        in_shape: Tuple[List[int]] = None,
    ):
        super(Block, self).__init__()
        if in_shape is not None:
            self.in_shape = in_shape
        self.name = name
        self.layers = ModuleDict()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def __getitem__(self, key: str) -> BlockType:
        return self.layers[key]

    def __setitem__(self, key: str, value: BlockType):
        if key in self.layers:
            raise AttributeError(f"Layer with name {key} already exists")
        self.layers[key] = value
        assert self.is_valid()

    def is_valid(self):
        if hasattr(self, "in_shape"):
            try:
                self.out_shape(self.in_shape)
            except Exception as e:
                raise e(f"Data shape mismatch while adding layer {key} to {self.name}")
        return True

    def out_shape(self, in_shape):
        x = torch.rand(1, *in_shape)
        for _, layer in self.layers.items():
            x = layer(x)
        return x.shape[1:]

    def __repr__(self):
        return "\n".join(
            [f"{k} : {v.__class__.__name__}" for k, v in self.layers.items()]
        )


if __name__ == "__main__":
    from torch.nn import ReLU

    block = Block(name="this_is_a_block")
    block["cnn_0"] = nn.Conv2d(1, 32, 3, 1, 1)
    block["activation_0"] = ReLU()
    block["cnn_1"] = nn.Conv2d(32, 64, 3, 1, 1)
    block["activation_1"] = ReLU()

    print("===" * 10)
    print("Block")
    print("===" * 10)
    print(block)
    print("===" * 10)
    print(block.out_shape([1, 32, 32]))
    print("===" * 10)
    print(block.layers)
