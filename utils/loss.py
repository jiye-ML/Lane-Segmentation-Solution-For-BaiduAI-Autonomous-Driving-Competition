import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MyLoss(nn.Module):

  def __init__(self, nbclasses):

    super(MyLoss, self).__init__()
    self.loss = None
    self.nbclasses = nbclasses


class MyCrossEntropyLoss(MyLoss):

  def __init__(self, nbclasses):
    super(MyCrossEntropyLoss, self).__init__(nbclasses)
    self.loss = nn.CrossEntropyLoss(reduction="mean")

  def forward(self, predicts, labels):

    if predicts.dim() > 2:
      # permute to (n, h, w, c)
      predicts = predicts.permute((0, 2, 3, 1))
      # reshape to (-1, num_classes)  每个像素在每种分类上都有一个概率
      predicts = predicts.reshape((-1, self.nbclasses))

    labels = labels.flatten()

    return self.loss(predicts, labels)


class MyDiceLoss(MyLoss):

  def __init__(self, nbclasses):
    super(MyDiceLoss, self).__init__(nbclasses)
    self.loss = DiceLoss()

  def forward(self, predicts, labels):

    labels_one_hot = make_one_hot(labels.reshape((-1, 1)), self.nbclasses)

    # reshape to (-1, num_classes)  每个像素在每种分类上都有一个概率
    predicts = predicts.reshape((-1, self.nbclasses))

    return self.loss(predicts, labels_one_hot.to(labels.device))


class FocalLoss(nn.Module):

  def __init__(self, gamma=0, alpha=None, size_average=True):
    super(FocalLoss, self).__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.alpha = torch.tensor([alpha, 1 - alpha])
    self.size_average = size_average

  def forward(self, inputs, target):
    if inputs.dim() > 2:
      inputs = inputs
      inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
      inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
      inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
    target = target.view(-1, 1)

    logpt = F.log_softmax(inputs, dim=1)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = logpt.exp()

    if self.alpha is not None:
      if self.alpha.type() != inputs.data.type():
        self.alpha = self.alpha.type_as(inputs.data)
      at = self.alpha.gather(0, target.view(-1))
      logpt = logpt * at
    # mask = mask.view(-1)
    loss = -1 * (1 - pt) ** self.gamma * logpt  # mask
    if self.size_average:
      return loss.mean()
    else:
      return loss.sum()


class BinaryDiceLoss(nn.Module):
  """Dice loss of binary class
  Args:
      smooth: A float number to smooth loss, and avoid NaN error, default: 1
      p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
      predict: A tensor of shape [N, *]
      target: A tensor of shape same with predict
      reduction: Reduction method to apply, return mean over batch if 'mean',
          return sum if 'sum', return a tensor of shape [N,] if 'none'
  Returns:
      Loss tensor according to arg reduction
  Raise:
      Exception if unexpected reduction
  """

  def __init__(self, smooth=1, p=2, reduction='mean'):
    super(BinaryDiceLoss, self).__init__()
    self.smooth = smooth
    self.p = p
    self.reduction = reduction

  def forward(self, predict, target):
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
    den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

    loss = 1 - num / den

    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    elif self.reduction == 'none':
      return loss
    else:
      raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
  """Dice loss, need one hot encode input
  Args:
      weight: An array of shape [num_classes,]
      ignore_index: class index to ignore
      predict: A tensor of shape [N, C, *]
      target: A tensor of same shape with predict
      other args pass to BinaryDiceLoss
  Return:
      same as BinaryDiceLoss
  """

  def __init__(self, weight=None, ignore_index=None, **kwargs):
    super(DiceLoss, self).__init__()
    self.kwargs = kwargs
    self.weight = weight
    self.ignore_index = ignore_index

  def forward(self, predict, target):
    assert predict.shape == target.shape, 'predict & target shape do not match'
    dice = BinaryDiceLoss(**self.kwargs)
    total_loss = 0

    for i in range(target.shape[1]):
      if i != self.ignore_index:
        dice_loss = dice(predict[:, i], target[:, i])
        if self.weight is not None:
          assert self.weight.shape[0] == target.shape[1], \
            'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
          dice_loss *= self.weights[i]
        total_loss += dice_loss

    return total_loss / target.shape[1]


def make_one_hot(x, num_classes):
  """Convert class index tensor to one hot encoding tensor.
  Args:
       x: A tensor of shape [N, 1, *]
       num_classes: An int of number of class
  Returns:
      A tensor of shape [N, num_classes, *]
  """
  shape = np.array(x.shape)
  shape[1] = num_classes
  shape = tuple(shape)
  result = torch.zeros(shape)
  result = result.scatter_(1, x.cpu().long(), 1)

  return result


def create_loss(
    predicts: torch.Tensor,
    labels: torch.Tensor,
    num_classes,
    cal_miou=True
):
  """
  创建loss
  @param predicts: shape=(n, c, h, w)
  @param labels: shape=(n, h, w) or shape=(n, 1, h, w)
  @param num_classes: int should equal to channels of predicts
  @param cal_miou:
  @return: loss, mean_iou
  """
  # BCE with DICE
  bce_loss = MyCrossEntropyLoss(num_classes)(predicts, labels)

  # 将labels做one_hot处理，得到的形状跟predicts相同
  dice_loss = MyDiceLoss(num_classes)(predicts, labels)

  loss = bce_loss + dice_loss

  return loss

