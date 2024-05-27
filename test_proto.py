r""" Hypercorrelation Squeeze testing code """
import argparse

import torch.nn.functional as F
import torch.nn as nn
import torch

from model.cdmd import CDMDNet
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset


def test(model, dataloader, nshot):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        logit_mask = model.predict_mask(batch)
        pred_mask=(logit_mask>0.5).float().squeeze(1)

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],batch['org_query_imsize'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../../data')
    parser.add_argument('--benchmark', type=str, default='coco', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='log_queryformer_r50_5s_coco_fold0')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='pretrained/coco/fold0/r50_1shot.pt')
    parser.add_argument('--layers', type=int, default=50)
    parser.add_argument('--reduce_dim', type=int, default=64)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--visualize', default=True)
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = CDMDNet(layers=args.layers, shot=args.shot, reduce_dim=args.reduce_dim)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize,'./vis/')

    # Dataset initialization
    FSSDataset.initialize(img_size=473, datapath=args.datapath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', args.shot)

    # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.shot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
