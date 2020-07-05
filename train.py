import os
import time
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from dataset import ICDARDataSet
from model import EAST
from loss import Loss
from evaluation import evaluate_batch


def train(config):
    tb_writer = SummaryWriter(config.out)

    train_dataset = ICDARDataSet(config.train_data_path)
    file_num = train_dataset.get_num_of_data()
    train_loader = data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                   num_workers=config.num_workers, drop_last=True)
    criterion = Loss()
    model = EAST()

    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[config.epoch // 2, config.epoch//2 +
    # config.epoch//4, config.epoch//2], gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, min_lr=1e-5)

    for epoch in range(config.epoch):
        model.train()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in tqdm(enumerate(train_loader), desc='Training...'):
            img = img.to(device)
            gt_score, gt_geo, ignored_map = gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
            pred_score, pred_geo = model(img)
            total_loss, classify_loss, angle_loss, iou_loss, geo_loss = criterion(gt_score, pred_score, gt_geo,
                                                                                  pred_geo, ignored_map)
            if i % 20 == 1:
                tb_writer.add_scalar('train/loss', total_loss, epoch * len(train_dataset) + i)
                tb_writer.add_scalar('train/classify_loss', classify_loss, epoch * len(train_dataset) + i)
                tb_writer.add_scalar('train/angle_loss', angle_loss, epoch * len(train_dataset) + i)
                tb_writer.add_scalar('train/iou_loss', iou_loss, epoch * len(train_dataset) + i)
                tb_writer.add_scalar('train/geo_loss', geo_loss, epoch * len(train_dataset) + i)

            epoch_loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / int(file_num / config.train_batch_size)
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss, time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)
        scheduler.step(epoch_loss)

        if (epoch + 1) % config.save_interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(config.out, 'model_epoch_{}.pth'.format(epoch + 1)))

        if config.eval_interval > 0 and (epoch + 1) % config.eval_interval == 0:
            _, eval_result = evaluate_batch(model, config)
            print(eval_result)
            tb_writer.add_scalar('train/hmean', eval_result['hmean'], (epoch + 1) * len(train_dataset))
            tb_writer.add_scalar('train/precision', eval_result['precision'], (epoch + 1) * len(train_dataset))
            tb_writer.add_scalar('train/recall', eval_result['recall'], (epoch + 1) * len(train_dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='../merged/train')
    parser.add_argument('--eval_data_path', type=str, default='../ICDAR_2015/test')
    parser.add_argument('--out', type=str, default='pths')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=600)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=5)
    args = parser.parse_args()
    train(args)
