from torch import nn




class vgg16(nn.Module):
    def __init__(self, numClasses=10):
        super(vgg16, self).__init__()

        # 100% 还原特征提取层，也就是5层共13个卷积层
        # conv1 1/2
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2 1/4
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3 1/8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4 1/16
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5 1/32
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 但是至于后面的全连接层，根据实际场景，就得自行定义自己的FC层了。
        self.classifier = nn.Sequential(  # 定义自己的分类层
            # 原始模型vgg16输入image大小是224 x 224
            # 我们测试的自己模仿写的模型输入image大小是32 x 32
            # 大小是小了 7 x 7倍
            nn.Linear(in_features=512 * 1 * 1, out_features=256),  # 自定义网络输入后的大小。
            # nn.Linear(in_features=512 * 7 * 7, out_features=256),  # 原始vgg16的大小是 512 * 7 * 7 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=numClasses),
        )
    def forward(self, x):  # output: 32 * 32 * 3
            x = self.relu1_1(self.conv1_1(x))  # output: 32 * 32 * 64
            x = self.relu1_2(self.conv1_2(x))  # output: 32 * 32 * 64
            x = self.pool1(x)  # output: 16 * 16 * 64

            x = self.relu2_1(self.conv2_1(x))
            x = self.relu2_2(self.conv2_2(x))
            x = self.pool2(x)

            x = self.relu3_1(self.conv3_1(x))
            x = self.relu3_2(self.conv3_2(x))
            x = self.relu3_3(self.conv3_3(x))
            x = self.pool3(x)

            x = self.relu4_1(self.conv4_1(x))
            x = self.relu4_2(self.conv4_2(x))
            x = self.relu4_3(self.conv4_3(x))
            x = self.pool4(x)

            x = self.relu5_1(self.conv5_1(x))
            x = self.relu5_2(self.conv5_2(x))
            x = self.relu5_3(self.conv5_3(x))
            x = self.pool5(x)

            x = x.view(x.size(0), -1)
            output = self.classifier(x)
            return output
