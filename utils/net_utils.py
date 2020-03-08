"""
@description: 跟网络模型有关的函数库
"""

"""
import
"""
from os.path import dirname, abspath
import sys
import model  # 本地

sys.path.append(dirname(dirname(abspath(__file__))))


def create_net(config, net_name='unet'):
  """
  创建网络
  :param config: 数据网路参数
  :param net_name: 网络类型，可选 unet , deeplabv3p
  """
  if net_name == 'unet':
    net = model.ResNetUNet(config)
  elif net_name == 'deeplabv3p':
    net = model.DeeplabV3Plus(config)
  else:
    raise ValueError('Not supported net_name: {}'.format(net_name))

  return net


def adjust_learning_rate(optimizer, lr_strategy, epoch, iteration, epoch_size):
  """
  根据给定的策略调整学习率
  @param optimizer: 优化器
  @param lr_strategy: 策略，一个二维数组，第一维度对应epoch，第二维度表示在一个epoch内，若干阶段的学习率
  @param epoch: 当前在第几号epoch
  @param iteration: 当前epoch内的第几次迭代
  @param epoch_size: 当前epoch的总迭代次数
  """
  assert epoch < len(lr_strategy), 'lr strategy unconvering all epoch'
  batch = epoch_size // len(lr_strategy[epoch])
  lr = lr_strategy[epoch][-1]
  for i in range(len(lr_strategy[epoch])):
    if iteration < (i + 1) * batch:
      lr = lr_strategy[epoch][i]
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      break
  return lr
