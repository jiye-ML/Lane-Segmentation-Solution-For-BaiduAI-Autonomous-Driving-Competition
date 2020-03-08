from tqdm import tqdm
import torch
import os
import torch.nn.functional as F

import dataset
import utils
from config import ConfigTrain


def train_epoch(
    net,
    epoch,
    data_loader,
    optimizer,
    input_file,
    device,
    config,
    epoch_size
):

  net.train()

  total_loss = 0.0
  dataprocess = tqdm(data_loader)
  for iteration, batch_item in enumerate(dataprocess):

    image, mask = batch_item['image'], batch_item['mask']
    image = image.to(device)
    mask = mask.to(device)

    lr = utils.adjust_learning_rate(
      optimizer,
      config.LR_STRATEGY,
      epoch,
      iteration,
      epoch_size
    )

    optimizer.zero_grad()

    # forward
    out = net(image)
    out = F.softmax(out, dim=1)

    # loss
    loss, _ = utils.create_loss(
      out,
      mask,
      config.NUM_CLASSES,
      cal_miou=False
    )

    total_loss += loss.item()

    # 反向传播
    loss.backward()

    # 学习
    optimizer.step()

    # 界面显示
    dataprocess.set_description_str("epoch:{}".format(epoch))
    dataprocess.set_postfix_str("loss:{:.4f}".format(loss.item()))

  input_file.write(
    "Epoch:{}, loss is {:.4f} \n".format(epoch, total_loss / len(data_loader))
  )
  input_file.flush()


def test(
    net,
    epoch,
    data_loader,
    result_file,
    config,
    device
):

  net.eval()

  if config.DEVICE.find('cuda') != -1:
    torch.cuda.empty_cache()  # 回收缓存的显存

  result = {
    "TP": {i: 0 for i in range(8)},
    "TA": {i: 0 for i in range(8)}
  }

  total_loss = 0.0
  mean_iou = 0.0
  confusion_matrix = None
  dataprocess = tqdm(data_loader)

  for batch_item in dataprocess:

    image, mask = batch_item['image'], batch_item['mask']
    image = image.to(device)
    mask = mask.to(device)

    out = net(image)
    out = F.softmax(out, dim=1)

    loss, iou = utils.create_loss(
      out, mask, config.NUM_CLASSES, cal_miou=True
    )
    total_loss += loss.item()
    mean_iou += iou.item()

    # 计算每个类别的混淆矩阵
    confusion_matrix = utils.compute_confusion_matrix(out, mask, result)

    dataprocess.set_description_str("epoch:{}".format(epoch))
    dataprocess.set_postfix_str(
      "mask_loss:{:.4f}, iou:{:.4f}".format(loss, iou)
    )

  mean_iou = mean_iou / len(dataprocess)
  result_file.write("Epoch:{} \n".format(epoch, mean_iou))
  for i in range(8):
    result_string = "{}: {:.4f} \n".format(
      i,
      confusion_matrix["TP"][i] / confusion_matrix["TA"][i]
    )
    print(result_string)
    result_file.write(result_string)

  # message
  info = "Epoch:{}, mean loss is {:.4f} mean_iou:{:.4f} \n".format(
    epoch, total_loss / len(data_loader), mean_iou
  )
  result_file.write(info)
  result_file.flush()


def main():
  cfg = ConfigTrain()

  print('Pick device: ', cfg.DEVICE)
  device = torch.device(cfg.DEVICE)

  # save path
  if not os.path.exists(cfg.LOG_ROOT):
    os.makedirs(cfg.LOG_ROOT)

  # input csv
  train_result_csv_path = open(
    os.path.join(cfg.LOG_ROOT, "train.csv"),
    'w'
  )
  test_result_csv_path = open(
    os.path.join(cfg.LOG_ROOT, "test.csv"),
    'w'
  )

  kwargs = {}
  if torch.cuda.is_available():
    kwargs = {'num_workers': 8, 'pin_memory': True}

  # train data
  train_data_loader, train_data_size = dataset.train_data_generator(
    cfg.DATA_LIST_ROOT, cfg.BATCH_SIZE, **kwargs
  )

  # data for val
  test_data_loader = dataset.test_data_generator(
    cfg.DATA_LIST_ROOT, 1, **kwargs
  )

  # 网络
  print('Generating net: ', cfg.NET_NAME)
  net = utils.create_net(cfg, net_name=cfg.NET_NAME)

  # 优化
  base_optimizer = utils.RAdam(net.parameters(), lr=cfg.BASE_LR)
  optimizer = utils.Lookahead(base_optimizer)

  # 加载与训练模型
  start_epoch = 0
  if cfg.PRETRAIN:
    print('Load pretrain weights: ', cfg.PRETRAINED_WEIGHTS)
    checkpoint = torch.load(cfg.PRETRAINED_WEIGHTS, map_location='cpu')
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
  net.to(device)

  # 一个轮次包含的迭代次数
  epoch_size = train_data_size / cfg.BATCH_SIZE

  # train
  for epoch in range(start_epoch, cfg.EPOCH_NUM):

    train_epoch(
      net=net,
      epoch=epoch,
      data_loader=train_data_loader,
      optimizer=optimizer,
      input_file=train_result_csv_path,
      device=device,
      config=cfg,
      epoch_size=epoch_size
    )

    if epoch % 5 == 0:

      # test
      test(net, epoch, test_data_loader, test_result_csv_path, cfg, device)

      # save
      _save_path = os.path.join(
        cfg.LOG_ROOT,
        "laneNet{}.pth.tar".format(epoch)
      )
      torch.save(
        {
          'state_dict': net.state_dict(),
          'epoch': epoch + 1,
          'optimizer': optimizer.state_dict()
        },
        _save_path
      )

  train_result_csv_path.close()
  test_result_csv_path.close()


if __name__ == "__main__":
  main()
