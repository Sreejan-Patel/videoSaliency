import argparse
import os
import time

import torch
from PIL import Image
import cv2 as cv

from vinet_model import ViNetModel
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--weight_file', default='', type=str, help='path to pretrained model state dict file')
parser.add_argument('--type', default='vinet_conv', type=str, help='type of model')
parser.add_argument('--test_data_path',
                    default='/scratch/smai/test',
                    type=str,
                    help='path to testing data')
parser.add_argument('--output_path', default='./result', type=str, help='path for output files')


def main():
    args = parser.parse_args()

    # set constants
    len_temporal = 32

    # set input and output path strings
    file_weight = args.weight_file
    path_input = args.test_data_path
    path_output = args.output_path
    path_output = os.path.join(path_output, time.strftime("%m-%d_%H-%M-%S"))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = ViNetModel(model_type=args.type)
    model.load_state_dict(torch.load(file_weight))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    list_input_data = [d for d in os.listdir(path_input) if os.path.isdir(os.path.join(path_input, d))]
    list_input_data.sort()

    for data_name in list_input_data:
        print(f'Processing {data_name}...')
        list_frames = [f for f in os.listdir(os.path.join(path_input, data_name, 'images')) if os.path.isfile(
            os.path.join(path_input, data_name, 'images', f)
        )]
        list_frames.sort()
        os.makedirs(os.path.join(path_output, data_name), exist_ok=True)

        if len(list_frames) < 2 * len_temporal - 1:
            print('Not enough frames in input clip!')
            return

        snippet = []
        for i in range(len(list_frames)):
            img = Image.open(os.path.join(path_input, data_name, 'images', list_frames[i])).convert('RGB')
            img_size = img.size
            img = torch_transform_image(img)

            snippet.append(img)

            if i >= len_temporal - 1:
                clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                clip = clip.permute((0, 2, 1, 3, 4))

                process_image(model, device, clip, data_name, list_frames[i], path_output, img_size)

                # process first (len_temporal - 1) frames
                if i < 2 * len_temporal - 2:
                    process_image(
                        model,
                        device,
                        torch.flip(clip, [2]),
                        data_name,
                        list_frames[i - len_temporal + 1],
                        path_output,
                        img_size
                    )

                del snippet[0]


def process_image(model, device, clip, data_name, frame_no, save_path, img_size):
    with torch.no_grad():
        pred = model(clip.to(device)).cpu().data[0]

    pred = pred.numpy()
    pred = cv.resize(pred, (img_size[0], img_size[1]))
    pred = blur(pred)

    save_image(pred, os.path.join(save_path, data_name, frame_no), normalize=True)


if __name__ == '__main__':
    main()
