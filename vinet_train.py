import argparse
import os
import time
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import *
from dataloader import DHF1KDataset
from loss import Loss
from vinet_model import ViNetModel

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path',
                    default='/scratch/smai/train',
                    type=str,
                    help='path to training data')
parser.add_argument('--validation_data_path',
                    default='/scratch/smai/val',
                    type=str,
                    help='path to validation data')
parser.add_argument('--type', default='vinet_conv', type=str, help='type of model')
parser.add_argument('--output_path', default='result', type=str, help='path for output files')
parser.add_argument('--S3D_weights_file', default='S3D_kinetics400.pt', type=str, help='path to S3D network weights file')
parser.add_argument('--model_val_path',default="vinet_model.pt", type=str)


def main():
    args = parser.parse_args()

    # set constants
    len_temporal = 32    # number of frames in operated clip
    batch_size = 8      # number of samples operated by the model at once
    epochs = 25

    # set input and output path strings
    path_train = args.train_data_path
    path_validate = args.validation_data_path
    path_output = args.output_path
    # path_output = os.path.join(path_output, time.strftime("%m-%d_%H-%M-%S"))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = ViNetModel(model_type=args.type)

    # load dataset
    train_dataset = DHF1KDataset(path_train, len_temporal)
    validation_dataset = DHF1KDataset(path_validate, len_temporal, mode='validate')

    # load the weight file for encoder network
    file_weight = args.S3D_weights_file
    if not os.path.isfile(file_weight):
        print('Invalid weight file for encoder network.')

    print(f'Loading encoder network weights from {file_weight}...')
    weight_dict = torch.load(file_weight)
    model_dict = model.backbone.state_dict()
    for name, param in weight_dict.items():
        if 'module' in name:
            name = '.'.join(name.split('.')[1:])
        if 'base.' in name:
            bn = int(name.split('.')[1])
            sn_list = [0, 5, 8, 14]
            sn = sn_list[0]
            if sn_list[1] <= bn < sn_list[2]:
                sn = sn_list[1]
            elif sn_list[2] <= bn < sn_list[3]:
                sn = sn_list[2]
            elif bn >= sn_list[3]:
                sn = sn_list[3]
            name = '.'.join(name.split('.')[2:])
            name = 'base%d.%d.' % (sn_list.index(sn) + 1, bn - sn) + name
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print(' size? ' + name, param.size(), model_dict[name].size())
        else:
            print(' name? ' + name)

    model.backbone.load_state_dict(model_dict)
    print(' Encoder network weights loaded!')


    # load model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model.to(device)

    # set parameters for training
    params = list(filter(lambda p: p.requires_grad, model.parameters())) 

    optimizer = torch.optim.Adam(params, lr=1e-4)
    criterion = Loss()

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    best_loss = None
    best_model = None
    for i in range(epochs):
        # train the model
        loss_train = train(model, train_loader, optimizer, criterion, i + 1, device)

        # validate the model
        with torch.no_grad():
            loss_val = validate(model, val_loader, criterion, i + 1, device)

            if loss_val[0] <= best_loss:
                best_loss = loss_val[0]
                best_model = model
                if torch.cuda.device_count() > 1:    
                    torch.save(model.module.state_dict(), args.model_val_path)
                else:
                    torch.save(model.state_dict(), args.model_val_path)
                



def train(model, loader, optimizer, criterion, epoch, device):
    print(f'\nStarting training model at epoch {epoch}\n')
    model.train()
    start_time = time.time()
    loss_sum, sim_sum, nss_sum, auc_sum = 0, 0, 0, 0
    num_samples = len(loader)

    for (idx, sample) in enumerate(loader):
        clips, gt, fixations = prepare_sample(sample, device, gt_to_device=True)
        optimizer.zero_grad()

        prediction = model(clips)
        assert prediction.size() == gt.size()

        loss, loss_sim, loss_nss = criterion(prediction, gt, fixations)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        # auc_sum += loss_auc.item()
        sim_sum += loss_sim.item()
        nss_sum += loss_nss.item()

    avg_loss = loss_sum / num_samples
    # avg_auc = auc_sum / num_samples
    avg_sim = sim_sum / num_samples
    avg_nss = nss_sum / num_samples
    print(f'\nepoch: {epoch}\n'
          f'loss: {avg_loss:.3f}\n'
          f'SIM: {avg_sim:.3f}\n'
          # f'AUC: {avg_auc:.3f}\n'
          f'NSS: {avg_nss:.3f}\n'
          f'training time: {((time.time() - start_time) / 60):.2f} minutes')
    return avg_loss, avg_sim, avg_nss


def validate(model, loader, criterion, epoch, device):
    print(f'\nStarting validating model at epoch {epoch}')
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        loss_sum, sim_sum, nss_sum, auc_sum = 0, 0, 0, 0
        num_samples = len(loader)

        for (idx, sample) in enumerate(loader):
            print(f' VAL: Processing sample {idx + 1}...')
            clips, gt, fixations = prepare_sample(sample, device, gt_to_device=False)

            prediction = model(clips)
            gt = gt.squeeze(0).numpy()
            prediction = prediction.cpu().squeeze(0).detach().numpy()
            prediction = cv.resize(prediction, (gt.shape[1], gt.shape[0]))
            prediction = blur(prediction).unsqueeze(0).cuda()
            gt = torch.FloatTensor(gt).unsqueeze(0).cuda()
            assert prediction.size() == gt.size()

            loss, loss_sim, loss_nss = criterion(prediction, gt, fixations)
            print(f'   loss: {loss.item():.3f}, SIM: {loss_sim:.3f}, NSS: {loss_nss:.3f}')
            loss_sum += loss.item()
            # auc_sum += loss_auc.item()
            sim_sum += loss_sim.item()
            nss_sum += loss_nss.item()

        avg_loss = loss_sum / num_samples
        # avg_auc = auc_sum / num_samples
        avg_sim = sim_sum / num_samples
        avg_nss = nss_sum / num_samples
        print(f'\nepoch: {epoch}\n'
              f'loss: {avg_loss:.3f}\n'
              f'SIM: {avg_sim:.3f}\n'
              # f'AUC: {avg_auc:.3f}\n'
              f'NSS: {avg_nss:.3f}\n'
              f'validation time: {((time.time() - start_time) / 60):.2f} minutes')
        return avg_loss, avg_sim, avg_nss


def prepare_sample(sample, device, gt_to_device):
    clips = sample[0]
    gt = sample[1]
    fixations = sample[2]
    clips = clips.to(device)
    clips = clips.permute((0, 2, 1, 3, 4))
    if gt_to_device:
        gt.to(device)
    return clips, gt, fixations


if __name__ == '__main__':
    main()
