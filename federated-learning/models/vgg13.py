import math

from torch import nn


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 64 * 32 * 32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64 * 32 * 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64 * 16 * 16
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128 * 16 * 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 * 16 * 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128 * 8 * 8
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 * 8 * 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 * 8 * 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 * 8 * 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 256 * 4 * 4
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 512 * 4 * 4
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512 * 4 * 4
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 512 * 4 * 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512 * 2 * 2
        )
        self.fc_layer1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 2 * 2, 4096),  # 1 * 4096
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 1 * 4096
            nn.ReLU()
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(4096, 10),  # 1 * num_class
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = self.conv_layer4(output)
        output = output.view(-1, 512 * 2 * 2)
        output = self.fc_layer1(output)
        output = self.fc_layer2(output)
        output = self.fc_layer3(output)
        return output