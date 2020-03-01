import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):

  def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
    super(Block, self).__init__()

    self.conv1 = nn.Conv2d(
      in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride
    )
    self.bn1 = nn.BatchNorm2d(out_ch)
    self.relu1 = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.relu1(self.bn1(self.conv1(x)))  # conv --> BN --> RELU
    return out


class ResBlock(nn.Module):

  def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):

    super(ResBlock, self).__init__()

    self.bn1 = nn.BatchNorm2d(in_ch)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(
      in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride
    )

  def forward(self, x):
    out = self.conv1(self.relu1(self.bn1(x)))   # BN --> RELU --> conv
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_chans, out_chans):
    """
    conv1 (c=1/4)output --> conv3x3 (c=1/4)output --> conv1 (c=1)output
    :param in_chans:
    :param out_chans:
    """

    super(Bottleneck, self).__init__()

    assert out_chans % 4 == 0

    # output channel = 1/4
    self.block1 = ResBlock(
      in_chans,
      int(out_chans / 4),
      kernel_size=1,
      padding=0
    )

    # output channel = 1/4
    self.block2 = ResBlock(
      int(out_chans / 4),
      int(out_chans / 4),
      kernel_size=3,
      padding=1
    )

    # output channel
    self.block3 = ResBlock(
      int(out_chans / 4),
      out_chans,
      kernel_size=1,
      padding=0
    )

  def forward(self, x):
    identity = x
    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    out += identity
    return out


class DownBottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_chans, out_chans, stride=2):

    super(DownBottleneck, self).__init__()

    assert out_chans % 4 == 0

    # 1/2
    self.block1 = ResBlock(
      in_chans,
      int(out_chans / 4),
      kernel_size=1,
      padding=0,
      stride=stride
    )

    # 1/4
    self.conv1 = nn.Conv2d(
      in_chans,
      out_chans,
      kernel_size=1,
      padding=0,
      stride=stride
    )

    self.block2 = ResBlock(
      int(out_chans / 4),
      int(out_chans / 4),
      kernel_size=3,
      padding=1
    )
    self.block3 = ResBlock(
      int(out_chans / 4),
      out_chans,
      kernel_size=1,
      padding=0
    )

  def forward(self, x):
    identity = self.conv1(x)
    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    out += identity
    return out


def make_layers(in_channels, layer_list, name="vgg"):
  layers = []
  if name == "vgg":
    for v in layer_list:
      layers += [Block(in_channels, v)]
      in_channels = v
  elif name == "resnet":
    layers += [DownBottleneck(in_channels, layer_list[0])]  # 1/4
    in_channels = layer_list[0]
    for v in layer_list[1:]:
      layers += [Bottleneck(in_channels, v)]
      in_channels = v
  return nn.Sequential(*layers)


class Layer(nn.Module):

  def __init__(self, in_channels, layer_list, net_name):

    super(Layer, self).__init__()

    self.layer = make_layers(in_channels, layer_list, name=net_name)

  def forward(self, x):
    out = self.layer(x)
    return out


class ASPP(nn.Module):

  def __init__(self, in_chans, out_chans, rate=1):

    super(ASPP, self).__init__()

    self.branch1 = nn.Sequential(
      nn.Conv2d(
        in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True
      ),
      nn.BatchNorm2d(out_chans),
      nn.ReLU(inplace=True),
    )
    self.branch2 = nn.Sequential(
      nn.Conv2d(
        in_chans, out_chans, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True
      ),
      nn.BatchNorm2d(out_chans),
      nn.ReLU(inplace=True),
    )
    self.branch3 = nn.Sequential(
      nn.Conv2d(
        in_chans, out_chans, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True
      ),
      nn.BatchNorm2d(out_chans),
      nn.ReLU(inplace=True),
    )
    self.branch4 = nn.Sequential(
      nn.Conv2d(
        in_chans, out_chans, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True
      ),
      nn.BatchNorm2d(out_chans),
      nn.ReLU(inplace=True),
    )

    # 全局特征

    self.branch5_avg = nn.AdaptiveAvgPool2d(1)
    self.branch5_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)
    self.branch5_bn = nn.BatchNorm2d(out_chans)
    self.branch5_relu = nn.ReLU(inplace=True)

    # 输出特征前的学习
    self.conv_cat = nn.Sequential(
      nn.Conv2d(out_chans * 5, out_chans, 1, 1, padding=0, bias=True),
      nn.BatchNorm2d(out_chans),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    b, c, h, w = x.size()

    # 采集输入不同尺度的特征
    conv1x1 = self.branch1(x)
    conv3x3_1 = self.branch2(x)
    conv3x3_2 = self.branch3(x)
    conv3x3_3 = self.branch4(x)

    # 采集数据的全局特征
    global_feature = self.branch5_avg(x)  # 变为 [1, 1]
    global_feature = self.branch5_relu(
      self.branch5_bn(self.branch5_conv(global_feature))
    )
    global_feature = F.interpolate( # 上采样的 [h, w]
      global_feature, (h, w), None, 'bilinear', True
    )

    # 不同尺度特征融合
    feature_cat = torch.cat(
      [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature],
      dim=1
    )

    result = self.conv_cat(feature_cat)

    return result
