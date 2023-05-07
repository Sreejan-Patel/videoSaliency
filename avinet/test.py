import argparse
import os
import time
import sys
import os
import numpy as np
import cv2 as cv

import torch
from os.path import join
import torchaudio
from PIL import Image

from AViNet.avinet_model import AViNetModel
from util.utils import torch_transform_image, save_image, blur

parser = argparse.ArgumentParser()
parser.add_argument('weight_file', default='', type=str, help='path to pretrained model state dict file')
parser.add_argument('--type', default='vinet_conv', type=str, help='type of model')
parser.add_argument('--dataset', default='DIEM', type=str, help='dataset name')
parser.add_argument('--test_data_path',
                    default='/scratch/sound_data',
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
    dataset = args.dataset
    path_output = os.path.join(path_output, time.strftime("%m-%d_%H-%M-%S"))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    model = AViNetModel(model_type=args.type)
    model.load_state_dict(torch.load(file_weight))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    list_input_data = []
    if args.dataset=='DIEM':
        file_name = 'DIEM_list_test_fps.txt'
    else:
        file_name = '{}_list_test_{}_fps.txt'.format(args.dataset, args.split)
	

    with open(join(path_input, 'fold_lists', file_name), 'r') as f:
        for line in f.readlines():
            name = line.split(' ')[0].strip()
            list_input_data.append(name)
    list_input_data.sort()
    
    audiodata = make_dataset(
			join(path_input, 'fold_lists', file_name), 
			join(path_input, 'video_audio', dataset),
			join(path_input, 'annotations', dataset)
		)
    
    for data_name in list_input_data:
        print(f'Processing {data_name}...')
        list_frames = [f for f in os.listdir(os.path.join(path_input, 'video_frames', dataset, data_name)) if os.path.isfile(
            os.path.join(path_input, 'video_frames', dataset, data_name, f)
        )]
        list_frames.sort()
        os.makedirs(os.path.join(path_output, data_name), exist_ok=True)

        if len(list_frames) < 2 * len_temporal - 1:
            print('Not enough frames in input clip!')
            return

        snippet = []
        for i in range(len(list_frames)):
            img = Image.open(os.path.join(path_input, 'video_frames', dataset, data_name, list_frames[i])).convert('RGB')
            img_size = img.size
            img = torch_transform_image(img)

            snippet.append(img)

            if i >= len_temporal-1:
                clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                clip = clip.permute((0,2,1,3,4))

                audio_feature = get_audio_feature(data_name, audiodata, args, i-len_temporal+1)
                process_image(model, device, clip, data_name, list_frames[i], path_output, img_size, audio_feature=audio_feature)
                
                if i < 2*len_temporal-2:
                    audio_feature = torch.flip(audio_feature, [2])
                    process_image(model, device, torch.flip(clip, [2]), data_name, list_frames[i-len_temporal+1], path_output, img_size, audio_feature=audio_feature)

                del snippet[0]

def process(model, clip, path_inpdata, dname, frame_no, args, img_size, audio_feature=None, device='cpu'):
	with torch.no_grad():
			smap = model(clip.to(device), audio_feature.to(device)).cpu().data[0]
	
	smap = smap.numpy()
	smap = cv.resize(smap, (img_size[0], img_size[1]))
	smap = blur(smap)

	save_image(smap, os.path.join(args.path_outdata, dname, frame_no), args.dataset)
 
def process_image(model, device, clip, data_name, frame_no, save_path, img_size, audio_feature=None):
    with torch.no_grad():
        pred = model(clip.to(device), audio_feature.to(device)).cpu().data[0]

    pred = pred.numpy()
    pred = cv.resize(pred, (img_size[0], img_size[1]))
    pred = blur(pred)

    save_image(pred, os.path.join(save_path, data_name, frame_no), normalize=True)
    
def read_sal_text(txt_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(txt_file,'r') as f:
		for line in f:
			word=line.strip().split()
			test_list['names'].append(word[0])
			test_list['nframes'].append(word[1])
			test_list['fps'].append(word[2])
	return test_list

def make_dataset(annotation_path, audio_path, gt_path, vox=False):
	data = read_sal_text(annotation_path)
	video_names = data['names']
	video_nframes = data['nframes']
	video_fps = data['fps']
	dataset = []
	audiodata= {}
	for i in range(len(video_names)):
		if i % 100 == 0:
			print('dataset loading [{}/{}]'.format(i, len(video_names)))

		n_frames = len(os.listdir(join(gt_path, video_names[i], 'maps')))
		if n_frames <= 1:
			print("Less frames")
			continue

		begin_t = 1
		end_t = n_frames

		audio_wav_path = os.path.join(audio_path,video_names[i],video_names[i]+'.wav')
		if not os.path.exists(audio_wav_path):
			print("Not exists", audio_wav_path)
			continue
			
		[audiowav,Fs] = torchaudio.load(audio_wav_path, normalization=False)
		audiowav = audiowav * (2 ** -23)
		n_samples = Fs/float(video_fps[i])
		starts=np.zeros(n_frames+1, dtype=int)
		ends=np.zeros(n_frames+1, dtype=int)
		starts[0]=0
		ends[0]=0
		for videoframe in range(1,n_frames+1):
			startemp=max(0,((videoframe-1)*(1.0/float(video_fps[i]))*Fs)-n_samples/2)
			starts[videoframe] = int(startemp)
			endtemp=min(audiowav.shape[1],abs(((videoframe-1)*(1.0/float(video_fps[i]))*Fs)+n_samples/2))
			ends[videoframe] = int(endtemp)

		audioinfo = {
			'audiopath': audio_path,
			'video_id': video_names[i],
			'Fs' : Fs,
			'wav' : audiowav,
			'starts': starts,
			'ends' : ends
		}

		audiodata[video_names[i]] = audioinfo

	return audiodata

def get_audio_feature(audioind, audiodata, args, start_idx):
	len_snippet = args.clip_size
	max_audio_Fs = 22050
	min_video_fps = 10
	max_audio_win = int(max_audio_Fs / min_video_fps * 32)

	audioexcer  = torch.zeros(1,max_audio_win)
	valid = {}
	valid['audio']=0

	if audioind in audiodata:

		excerptstart = audiodata[audioind]['starts'][start_idx+1]
		if start_idx+len_snippet >= len(audiodata[audioind]['ends']):
			print("Exceeds size", audioind)
			sys.stdout.flush()
			excerptend = audiodata[audioind]['ends'][-1]
		else:
			excerptend = audiodata[audioind]['ends'][start_idx+len_snippet]	
		try:
			valid['audio'] = audiodata[audioind]['wav'][:, excerptstart:excerptend+1].shape[1]
		except:
			pass
		audioexcer_tmp = audiodata[audioind]['wav'][:, excerptstart:excerptend+1]
		if (valid['audio']%2)==0:
			audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2))] = \
				torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
		else:
			audioexcer[:,((audioexcer.shape[1]//2)-(valid['audio']//2)):((audioexcer.shape[1]//2)+(valid['audio']//2)+1)] = \
				torch.from_numpy(np.hanning(audioexcer_tmp.shape[1])).float() * audioexcer_tmp
	audio_feature = audioexcer.view(1, 1,-1,1)
	return audio_feature


if __name__ == '__main__':
    main()
