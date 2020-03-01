from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss
from model.deeplabv3plus import DeeplabV3Plus
from model.unet import ResNetUNet
from config import Config


device_list = [0]
train_net = 'deeplabv3p'
nets = {'deeplabv3p': DeeplabV3Plus, 'unet': ResNetUNet}


def train_epoch(net, epoch, data_loader, optimizer, loss, input_file):

  net.train()

  total_mask_loss = 0.0
  dataprocess = tqdm(data_loader)
  for batch_item in dataprocess:

    image, mask = batch_item['image'], batch_item['mask']
    if torch.cuda.is_available():
      image = image.cuda(device=device_list[0])
      mask = mask.cuda(device=device_list[0])

    optimizer.zero_grad()

    # forward
    out = net(image)

    # loss
    mask_loss = loss(out, mask)
    total_mask_loss += mask_loss.item()

    # 反向传播
    mask_loss.backward()

    # 学习
    optimizer.step()

    # 界面显示
    dataprocess.set_description_str("epoch:{}".format(epoch))
    dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))

  input_file.write(
    "Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(data_loader))
  )
  input_file.flush()


def test(net, epoch, data_loader, result_file, loss):

  net.eval()

  result = {
    "TP": {i: 0 for i in range(8)},
    "TA": {i: 0 for i in range(8)}
  }

  total_mask_loss = 0.0
  dataprocess = tqdm(data_loader)

  for batch_item in dataprocess:

    image, mask = batch_item['image'], batch_item['mask']
    if torch.cuda.is_available():
      image = image.cuda(device=device_list[0])
      mask = mask.cuda(device=device_list[0])

    out = net(image)

    mask_loss = loss(out, mask)
    total_mask_loss += mask_loss.detach().item()

    pred = torch.argmax(F.softmax(out, dim=1), dim=1)
    result = compute_iou(pred, mask, result)

    dataprocess.set_description_str("epoch:{}".format(epoch))
    dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))

  result_file.write("Epoch:{} \n".format(epoch))
  for i in range(8):
    result_string = "{}: {:.4f} \n".format(i, result["TP"][i] / result["TA"][i])
    print(result_string)
    result_file.write(result_string)

  # message
  info = "Epoch:{}, mask loss is {:.4f} \n".format(
    epoch, total_mask_loss / len(data_loader)
  )
  result_file.write(info)
  result_file.flush()


def adjust_lr(optimizer, epoch):
  if epoch == 0:
    lr = 1e-3
  elif epoch == 2:
    lr = 1e-2
  elif epoch == 100:
    lr = 1e-3
  elif epoch == 150:
    lr = 1e-4
  else:
    return
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def main():
  lane_config = Config()

  # save path
  if not os.path.exists(lane_config.SAVE_PATH):
    os.makedirs(lane_config.SAVE_PATH)

  # input csv
  train_result_csv_path = open(
    os.path.join(lane_config.SAVE_PATH, "train.csv"),
    'w'
  )
  test_result_csv_path = open(
    os.path.join(lane_config.SAVE_PATH, "test.csv"),
    'w'
  )

  kwargs = {}
  if torch.cuda.is_available():
    kwargs = {'num_workers': 4, 'pin_memory': True}

  # train data
  train_dataset = LaneDataset(
    "train.csv",
    transform=transforms.Compose([
      ImageAug(),
      DeformAug(),
      ScaleAug(),
      CutOut(32, 0.5),
      ToTensor()
    ])
  )
  train_data_loader = DataLoader(
    train_dataset,
    batch_size=4 * len(device_list),
    shuffle=True,
    drop_last=True,
    **kwargs
  )

  # data for val
  val_dataset = LaneDataset(
    "val.csv",
    transform=transforms.Compose([ToTensor()])
  )
  val_data_loader = DataLoader(
    val_dataset,
    batch_size=2 * len(device_list),
    shuffle=False,
    drop_last=False,
    **kwargs
  )

  # net
  net = nets[train_net](lane_config)
  if torch.cuda.is_available():
    net = net.cuda(device=device_list[0])
    net = torch.nn.DataParallel(net, device_ids=device_list)

  # 优化
  optimizer = torch.optim.Adam(
    net.parameters(),
    lr=lane_config.BASE_LR,
    weight_decay=lane_config.WEIGHT_DECAY
  )

  # loss
  loss = MySoftmaxCrossEntropyLoss(lane_config.NUM_CLASSES)

  # resume model

  start_epoch = 0
  if lane_config.RESUME is not None:
    resume_path = os.path.join(lane_config.SAVE_PATH, lane_config.RESUME)
    if not os.path.isfile(resume_path):
      raise RuntimeError(
        "=> no checkpoint found at '{}'".format(resume_path)
      )
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch'] + 1
    if torch.cuda.is_available():
      net.module.load_state_dict(checkpoint['state_dict'])
    else:
      net.model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])

  # train
  for epoch in range(start_epoch, lane_config.EPOCHS):

    # adjust_lr(optimizer, epoch)
    train_epoch(
      net,
      epoch,
      train_data_loader,
      optimizer,
      loss,
      train_result_csv_path
    )

    # net.module.state_dict()
    if epoch % 2 == 0:

      # test
      test(net, epoch, val_data_loader, test_result_csv_path, loss)

      # save
      _save_path = os.path.join(
        os.getcwd(),
        lane_config.SAVE_PATH,
        "laneNet{}.pth.tar".format(epoch)
      )
      torch.save({'state_dict': net.state_dict()}, _save_path)

  train_result_csv_path.close()
  test_result_csv_path.close()


if __name__ == "__main__":
  main()
