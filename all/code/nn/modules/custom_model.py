import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(Attention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, self.num_heads, self.head_dim, -1)
        key = self.key(x).view(batch_size, self.num_heads, self.head_dim, -1)
        value = self.value(x).view(batch_size, self.num_heads, self.head_dim, -1)

        energy = torch.einsum('bhqd, bhkd -> bhqk', query, key) * self.scale
        attention = self.softmax(energy)
        out = torch.einsum('bhqk, bhvd -> bhqd', attention, value)
        out = out.contiguous().view(batch_size, channels, height, width)
        out = self.out_conv(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class CustomCNN(nn.Module):
    def __init__(self, dropout, num_heads=4, num_classes=1):
        super(CustomCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.res_conv = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.res_bn = nn.BatchNorm2d(256)

        self.attention = Attention(256, num_heads)
        self.se_block = SEBlock(256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten_dim = self._get_flatten_dim()

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 4 + num_classes)  # 4个坐标值加上类别预测

        self.dropout = nn.Dropout(dropout)

    def _get_flatten_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 28, 28)
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool1(F.relu(self.bn2(self.conv2(x))))
            x = self.pool1(F.relu(self.bn3(self.conv3(x))))
            x = self.pool1(F.relu(self.bn4(self.conv4(x))))
            x = self.adaptive_pool(x)
            flatten_dim = x.view(1, -1).size(1)
        return flatten_dim

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(F.relu(self.bn3(self.conv3(x))))
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))
        residual = x
        residual = F.relu(self.res_bn(self.res_conv(residual)))
        x = x + residual
        x = self.attention(x)
        x = self.se_block(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn5(self.fc1(x))))
        x = self.dropout(F.relu(self.bn6(self.fc2(x))))
        x = self.fc3(x)
        return x
