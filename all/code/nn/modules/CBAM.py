import torch
import torch.nn as nn



"""
通道注意力模型: 通道维度不变，压缩空间维度。该模块关注输入图片中有意义的信息。
1）假设输入的数据大小是(b,c,w,h)
2）通过自适应平均池化使得输出的大小变为(b,c,1,1)
3）通过2d卷积和sigmod激活函数后，大小是(b,c,1,1)
4）将上一步输出的结果和输入的数据相乘，输出数据大小是(b,c,w,h)。
"""


class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


"""
空间注意力模块：空间维度不变，压缩通道维度。该模块关注的是目标的位置信息。
1） 假设输入的数据x是(b,c,w,h)，并进行两路处理。
2）其中一路在通道维度上进行求平均值，得到的大小是(b,1,w,h)；另外一路也在通道维度上进行求最大值，得到的大小是(b,1,w,h)。
3） 然后对上述步骤的两路输出进行连接，输出的大小是(b,2,w,h)
4）经过一个二维卷积网络，把输出通道变为1，输出大小是(b,1,w,h)
4）将上一步输出的结果和输入的数据x相乘，最终输出数据大小是(b,c,w,h)。
"""


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))
