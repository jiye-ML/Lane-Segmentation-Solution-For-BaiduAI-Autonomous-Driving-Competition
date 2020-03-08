import numpy as np
import torch
import torch.nn.functional as F


def compute_confusion_matrix(pred, gt, result):
    """
    pred : [N, H, W]
    gt: [N, H, W]
    """
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    for i in range(8):
        single_gt = (gt == i)
        single_pred = (pred == i)
        temp_tp = np.sum(single_gt * single_pred)
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result


def compute_miou(predicts, labels, num_classes):
  """
  计算iou
  @param predicts: shape=(-1, classes)
  @param labels: shape=(-1, 1)
  @param num_classes: 类别数目
  """
  ious = torch.zeros(num_classes)
  predicts = F.softmax(predicts, dim=1)
  predicts = torch.argmax(predicts, dim=1, keepdim=True)
  for i in range(num_classes):
    intersect = torch.sum((predicts == i) * (labels == i))
    area = torch.sum(predicts == i) + torch.sum(labels == i) - intersect
    if area == 0 and intersect == 0:
      ious[i] = np.nan  # 忽略这种iou
    else:
      ious[i] = intersect.float() / area.float()
  return ious
