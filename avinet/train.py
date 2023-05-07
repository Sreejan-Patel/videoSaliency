import argparse
import os
import time
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from util import dataloader, loss, utils
from avinet import avinet_model

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',
                    default='/scratch/sound_data',
                    type=str,
                    help='path to data')
parser.add_argument('--type', default='vinet_conv', type=str, help='type of model')
parser.add_argument('--dataset', default='DIEM', type=str, help='dataset name')
parser.add_argument('--output_path', default='result', type=str, help='path for output files')
parser.add_argument('--vinet_weights_file', default='vinet_model.pt', type=str, help='path to vinet weights file')
parser.add_argument('--model_val_path',default="avinet_model.pt", type=str)


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

    model = avinet_model.AViNetModel(model_type=args.type)

    # load dataset
    train_dataset = dataloader.SoundDatasetLoader(path_train, len_temporal, args.dataset)
    val_dataset = dataloader.SoundDatasetLoader(path_validate, len_temporal, args.dataset, mode='test')

    # load the weight file for encoder network
    file_weight = args.vinet_weights_file
    print(f'Loading ViNet network weights from {file_weight}...')
    model.visual_model.load_state_dict(torch.load(args.load_weight))


    # load model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model.to(device)

    # set parameters for training
    params = list(filter(lambda p: p.requires_grad, model.parameters())) 

    optimizer = torch.optim.Adam(params, lr=1e-4)
    criterion = loss.Loss()

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

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
        print(f' TRAIN: Processing sample {idx + 1}...')
        clips, gt, fixations = prepare_sample(sample, device, gt_to_device=True)
        audio_feature = sample[2].to(device)
        optimizer.zero_grad()

        prediction = model(clips, audio_feature)
        assert prediction.size() == gt.size()

        loss, loss_sim, loss_nss = criterion(prediction, gt, fixations)
        loss.backward()
        optimizer.step()
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
            prediction = utils.blur(prediction).unsqueeze(0).cuda()
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
