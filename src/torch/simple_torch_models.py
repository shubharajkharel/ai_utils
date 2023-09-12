import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchinfo import summary


# Define a simple fully connected neural network
class SimpleTorchFCModel(nn.Module):
    def __init__(self, input_shape=(10,), output_size=2, hidden_size=30, device=None):
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

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        x = self.e(x)
        return x


class SimpleTorchCNNModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), output_size=10, device=None):
        super(SimpleTorchCNNModel, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(320, output_size)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        self.example_input = torch.randn(1, 1, 28, 28, device=device)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class SimpleTorchModel(nn.Module):
    def __init__(self, device=None):
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
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.example_input = torch.randn(1, 1, 28, 28, device=device)
        self.to(device)

    def forward(self, x):
        return self.model(x)

    # inputs = [Input(shape=mc.input_shape) for mc in config.mode_configs]


if __name__ == "__main__":
    model_pytorch = SimpleTorchCNNModel()
    model_pytorch().summary()
    # summary(model_pytorch, input_size=(1, 1, 28, 28))

    # model = SimpleTorchCNNModel()
    # print(model)
    # print(model(torch.randn(1, 1, 28, 28)))
