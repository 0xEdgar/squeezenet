import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class fireModule(nn.Module):
    def __init__(self, input_dims, s1x1 = 3, expand1x1=4, expand1x3=4): # output dimensions?
        super().__init__()
        self.squeeze = nn.Conv2d(input_dims, s1x1, 1)
        self.expand1 = nn.Conv2d(s1x1, expand1x1, 1)
        self.expand3 = nn.Conv2d(s1x1, expand1x3, 3)

    def forward(x):
        x = F.relu(self.squeeze(x))
        x = torch.cat([
        F.relu(self.expand1(x)),
        F.relu(self.expand3(x))
        ])
        return x

class SqueezeNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super().init()
        self.features = nn.Sequential(
        nn.Conv2d(3,96,7),
        nn.MaxPool2d(kernel_size = 3, stride = 2),
        fireModule(96, 16, 64, 64),
        fireModule(128, 16, 64, 64),
        fireModule(128, 32,128, 128),
        nn.MaxPool2d(kernel_size = 3, stride = 2),
        fireModule(256, 3, 128, 128),
        fireModule(256, 48, 192, 192),
        fireModule(384, 48, 192, 192),
        fireModule(384, 64, 256, 256),
        nn.MaxPool2d(kernel_size=3, stride = 2),
        fireModule(512, 64, 256, 256),
        nn.Conv2d(512, 1000, 1, 1),
        nn.AvgPool2d(13, 1)
        )
    def forward(self, x):
        x = self.features(x)
        return x
