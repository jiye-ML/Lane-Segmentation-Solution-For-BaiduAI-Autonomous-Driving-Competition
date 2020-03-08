import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.process_labels import encode_labels, decode_labels, decode_color_labels
from dataset import custom_transforms


class LaneDataset(Dataset):

  def __init__(self, data_frame, transform=None):
    super(LaneDataset, self).__init__()
    self.data = data_frame
    self.images = self.data["image"]
    self.labels = self.data["label"]

    self.transform = transform

  def __len__(self):
    return self.labels.shape[0]

  def __getitem__(self, idx):
    ori_image = cv2.imread(self.images[idx])
    ori_mask = cv2.imread(self.labels[idx], cv2.IMREAD_GRAYSCALE)
    train_img, train_mask = custom_transforms.crop_resize_data(
      ori_image, ori_mask
    )

    # Encode
    train_mask = encode_labels(train_mask)
    sample = [train_img.copy(), train_mask.copy()]
    if self.transform:
      sample = self.transform(sample)

    return sample


def expand_resize_data(
    prediction=None,
    submission_size=(3384, 1710),
    offset=690
):
  pred_mask = decode_labels(prediction)
  expand_mask = cv2.resize(
    pred_mask,
    (submission_size[0], submission_size[1] - offset),
    interpolation=cv2.INTER_NEAREST
  )
  submission_mask = np.zeros(
    (submission_size[1], submission_size[0]), dtype='uint8'
  )
  submission_mask[offset:, :] = expand_mask
  return submission_mask


def expand_resize_color_data(prediction=None, submission_size=(3384, 1710), offset=690):
  color_pred_mask = decode_color_labels(prediction)
  color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0))
  color_expand_mask = cv2.resize(
    color_pred_mask,
    (submission_size[0], submission_size[1] - offset),
    interpolation=cv2.INTER_NEAREST
  )
  color_submission_mask = np.zeros(
    (submission_size[1], submission_size[0], 3),
    dtype='uint8'
  )
  color_submission_mask[offset:, :, :] = color_expand_mask
  return color_submission_mask


def train_data_generator(data_list_root, batch_size, **kwargs):
  _data_frame = pd.read_csv(
    os.path.join(data_list_root, 'train.csv')
  )
  train_dataset = LaneDataset(
    _data_frame,
    transform=transforms.Compose([
      custom_transforms.ImageAug(),
      custom_transforms.DeformAug(),
      custom_transforms.ScaleAug(),
      custom_transforms.CutOut(32, 0.5),
      custom_transforms.ToTensor()
    ])
  )
  train_data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    **kwargs
  )

  return train_data_loader, len(_data_frame)


def test_data_generator(data_list_root, batch_size, **kwargs):
  _data_frame = pd.read_csv(
    os.path.join(data_list_root, 'val.csv')
  )
  test_dataset = LaneDataset(
    _data_frame,
    transform=transforms.Compose([
      custom_transforms.ToTensor()
    ])
  )
  test_data_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs
  )

  return test_data_loader
