r""" MSDNet training (validation) code """
import argparse
import os
import torch.optim as optim
import torch.nn as nn

import torch

from model.msd import MSDNet
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import time

def train(epoch, model, dataloader, optimizer, training):
    r""" Train MSDNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.train() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    start_time = time.time()
    for idx, batch in enumerate(dataloader):
        if idx % 100 == 0 and idx>0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time: {elapsed_time}s")
            start_time = time.time()
            
        batch = utils.to_cuda(batch)
        if training:
            #print(10)
            logit_mask = model(batch['query_img'], batch['support_imgs'], batch['support_masks'])
            loss = model.compute_loss(logit_mask, batch['query_mask'])
        else:
            # print(11)
            logit_mask = model.predict_mask(batch)
        pred_mask=(logit_mask>0.5).float().squeeze(1)
        if training:
            #print(12)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        if training:
            # print(13)
            average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=100)
        else:
            average_meter.update(area_inter, area_union, batch['class_id'],loss=None)
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=100)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='MSDNet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../data')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss', 'new_coco'])
    parser.add_argument('--logpath', type=str, default='log_MSDNet_r50_coco0')
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=30)
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--layers', type=int, default=50)
    parser.add_argument('--reduce_dim', type=int, default=64)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot',type=int, default=1)
    args = parser.parse_args()
    Logger.initialize(args, training=True)

    # Model initialization
    model = MSDNet(layers=args.layers,shot=args.shot,reduce_dim=args.reduce_dim)
    Logger.log_params(model)
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    ### if you want to have L2 regularization, comment the above line and uncomment the following line:
    # optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr, "weight_decay": 0.001}])
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(img_size=473, datapath=args.datapath)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn',shot=args.shot)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, 1, args.nworker, args.fold, 'val',shot=args.shot)

    best_val_miou = float('-inf')
    best_val_loss = float('inf')

    for epoch in range(args.niter):
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)
        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
