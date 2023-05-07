import argparse
import os
import time

import torch
from PIL import Image
import cv2 as cv

from ViNet.vinet_model import ViNetModel
from loss import Loss
from dataloader import DHF1KDataset
from ViNet.train import prepare_sample
from torch.utils.data import DataLoader
from utils import  blur


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset used for validation')
parser.add_argument('--weights', type=str, default='weights/vinet_model.pt', help='Path to weights file.')
parser.add_argument('--criterion', type=str, default='metrics', help='Criterion which should be evaluated. (metrics, time)')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def evaluate():
    args = parser.parse_args()

    len_temporal = 32
    model = ViNetModel()
    model.load_state_dict(torch.load(args.weights))


    path_data_DHF1K = '/scratch/test'

    eval_criterion = args.criterion
    if eval_criterion == 'metrics':
        criterion = Loss(mode='evaluate')

    dataset_name = args.dataset
    if dataset_name == 'DHF1K':
        val_dataset = DHF1KDataset(path_data_DHF1K, len_temporal, mode='validate')
    else:
        print('Invalid dataset name.')
        return
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    with torch.no_grad():
        model.eval()
        sim_sum, nss_sum, auc_sum, cc_sum = 0, 0, 0, 0
        num_samples = len(loader)
        total_time, total_frames = 0, 0

        for (idx, sample) in enumerate(loader):
            start_time = time.time()
            print(f' Processing sample {idx + 1}...')
            clips, gt, fixations = prepare_sample(sample, device, gt_to_device=False)

            prediction = model(clips)
            if eval_criterion == 'metrics':
                gt = gt.squeeze(0).numpy()
                prediction = prediction.cpu().squeeze(0).detach().numpy()
                prediction = cv.resize(prediction, (gt.shape[1], gt.shape[0]))
                prediction = blur(prediction).unsqueeze(0).cuda()
                gt = torch.FloatTensor(gt).unsqueeze(0).cuda()
                # print(prediction.size())
                # print(gt.size())
                assert prediction.size() == gt.size()

                sim, nss, auc, cc = criterion(prediction, gt, fixations)
                print(f'  SIM: {sim.item():.3f}, NSS: {nss:.3f}, AUC: {auc.item():.3f}, CC: {cc.item():.3f}')
                sim_sum += sim.item()
                nss_sum += nss.item()
                auc_sum += auc.item()
                cc_sum += cc.item()
            elif eval_criterion == 'time':
                total_time += (time.time() - start_time)
                total_frames += len_temporal
                print(f'  time: {time.time() - start_time}')

        if eval_criterion == 'metrics':
            avg_sim = sim_sum / num_samples
            avg_nss = nss_sum / num_samples
            avg_auc = auc_sum / num_samples
            avg_cc = cc_sum / num_samples
            print(f'SIM: {avg_sim:.3f}\n'
                  f'NSS: {avg_nss:.3f}\n'
                  f'AUC: {avg_auc:.3f}\n'
                  f'CC: {avg_cc:.3f}\n')
        elif eval_criterion == 'time':
            print(f'total time: {total_time / 60} min\n'
                  f'FPS: {total_frames / total_time}')


if __name__ == '__main__':
    evaluate()
