import os
from os.path import join
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchaudio
import sys
import json
from scipy.io import loadmat
from utils import torch_transform_image

class DHF1KDataset(Dataset):
	def __init__(self, path_data, len_snippet, mode="train"):
		''' mode: train, validate '''
		self.path_data = path_data
		self.len_snippet = len_snippet
		self.mode = mode
		if self.mode == "train":
			self.video_names = os.listdir(path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(path_data,d,'images'))) for d in self.video_names]
		elif self.mode=="validate":
			self.list_num_frame = []
			for v in os.listdir(path_data):
				for i in range(0, len(os.listdir(os.path.join(path_data,v,'images'))) - self.len_snippet, 4*self.len_snippet):
					self.list_num_frame.append((v, i))

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
		elif self.mode == "validate":
			file_name, start_idx = self.list_num_frame[idx]
   
		path_clip = os.path.join(self.path_data, file_name, 'images')
		path_annotation = os.path.join(self.path_data, file_name, 'maps')
		path_fixation = os.path.join(self.path_data, file_name, 'fixation', 'maps')
		clip_img = []
		clip_fixation = []
		clip_annotation = []

		for i in range(self.len_snippet):
			img = Image.open(os.path.join(path_clip, f'{(start_idx + i + 1):04}.png')).convert('RGB')

			annotation = np.array(Image.open(os.path.join(path_annotation, f'{(start_idx + i + 1):04}.png')).convert('L'))
			annotation = annotation.astype(float)
			annotation = cv2.resize(annotation, (384, 224))
			if np.max(annotation) > 1.0:
				annotation = annotation / 255.0

			fixation = loadmat(os.path.join(path_fixation, f'{(start_idx + i + 1):04}.mat'))['I']
			fixation = resize_fixation(fixation)

			clip_img.append(torch_transform_image(img))
			clip_annotation.append(torch.FloatTensor(annotation))
			clip_fixation.append(torch.from_numpy(fixation.copy()))

		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
		clip_annotation = torch.FloatTensor(torch.stack(clip_annotation, dim=0))

		return clip_img, clip_annotation[-1], clip_fixation[-1]
        
		

class SoundDatasetLoader(Dataset):
	def __init__(self, len_snippet, path_data, dataset_name='DIEM', split=1, mode='train'):
		''' mode: train, validate, test'''
		self.path_data = path_data
		self.mode = mode
		self.len_snippet = len_snippet
		self.list_num_frame = []
		self.dataset_name = dataset_name
		if dataset_name=='DIEM':
			file_name = 'DIEM_list_{}_fps.txt'.format(mode)
		else:
			file_name = '{}_list_{}_{}_fps.txt'.format(dataset_name, mode, split)
		
		self.list_indata = []
		with open(join(self.path_data, 'fold_lists', file_name), 'r') as f:
			for line in f.readlines():
				name = line.split(' ')[0].strip()
				self.list_indata.append(name)

		self.list_indata.sort()	
		print(self.mode, len(self.list_indata))
		if self.mode=='train':
			self.list_num_frame = [len(os.listdir(os.path.join(path_data,'annotations', dataset_name, v, 'maps'))) for v in self.list_indata]
		
		elif self.mode == 'test' or self.mode == 'validate': 
			print("val set")
			for v in self.list_indata:
				frames = os.listdir(join(path_data, 'annotations', dataset_name, v, 'maps'))
				frames.sort()
				for i in range(0, len(frames)-self.len_snippet,  2*self.len_snippet):
					if self.check_frame(join(path_data, 'annotations', dataset_name, v, 'maps', 'eyeMap_%05d.jpg'%(i+self.len_snippet))):
						self.list_num_frame.append((v, i))

		max_audio_Fs = 22050
		min_video_fps = 10
		self.max_audio_win = int(max_audio_Fs / min_video_fps * 32)
		if self.mode=='validate':
			file_name = file_name.replace('val', 'test')
		json_file = '{}_fps_map.json'.format(self.dataset_name)
		self.audiodata = make_dataset(
				join(self.path_data, 'fold_lists', file_name), 
				join(self.path_data, 'video_audio', self.dataset_name),
				join(self.path_data, 'annotations', self.dataset_name),
			)

	def check_frame(self, path):
		img = cv2.imread(path, 0)
		return img.max()!=0

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		# print(self.mode)
		if self.mode == "train":
			video_name = self.list_indata[idx]
			while 1:
				start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
				if self.check_frame(join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps', 'eyeMap_%05d.jpg'%(start_idx+self.len_snippet))):
					break
				else:
					print("No saliency defined in train dataset")
					sys.stdout.flush()

		elif self.mode == "test" or self.mode == "val":
			(video_name, start_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, 'video_frames', self.dataset_name, video_name)
		path_annt = os.path.join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps')

		if self.use_sound:
			audio_feature = get_audio_feature(video_name, self.audiodata, self.len_snippet, start_idx)

		clip_img = []
		
		for i in range(self.len_snippet):
			img = Image.open(join(path_clip, 'img_%05d.jpg'%(start_idx+i+1))).convert('RGB')
			sz = img.size		
			clip_img.append(self.img_transform(img))
			
		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
		
		gt = np.array(Image.open(join(path_annt, 'eyeMap_%05d.jpg'%(start_idx+self.len_snippet))).convert('L'))
		gt = gt.astype('float')
		
		if self.mode == "train":
			gt = cv2.resize(gt, (384, 224))

		if np.max(gt) > 1.0:
			gt = gt / 255.0
		assert gt.max()!=0, (start_idx, video_name)
		if self.use_sound or self.use_vox:
			return clip_img, gt, audio_feature
		return clip_img, gt

def resize_fixation(fixation, width=384, height=224):
    out = np.zeros((height, width))
    height_sf = height / fixation.shape[0]  # height scale factor
    width_sf = width / fixation.shape[1]    # width scale factor

    coords = np.argwhere(fixation)
    for coord in coords:
        row = int(np.round(coord[0] * height_sf))
        col = int(np.round(coord[1] * width_sf))
        if row == height:
            row -= 1
        if col == width:
            col -= 1
        out[row, col] = 1

    return out

def read_sal_text(txt_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(txt_file,'r') as f:
		for line in f:
			word=line.strip().split()
			test_list['names'].append(word[0])
			test_list['nframes'].append(word[1])
			test_list['fps'].append(word[2])
	return test_list

def read_sal_text_dave(json_file):
	test_list = {'names': [], 'nframes': [], 'fps': []}
	with open(json_file,'r') as f:
		_dic = json.load(f)
		for name in _dic:
			# word=line.strip().split()
			test_list['names'].append(name)
			test_list['nframes'].append(0)
			test_list['fps'].append(float(_dic[name]))
	return test_list	

def make_dataset(annotation_path, audio_path, gt_path, json_file=None):
	if json_file is None:
		data = read_sal_text(annotation_path)
	else:
		data = read_sal_text_dave(json_file)
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

def get_audio_feature(audioind, audiodata, clip_size, start_idx):
	len_snippet = clip_size
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
	else:
		print(audioind, "not present in data")
	audio_feature = audioexcer.view(1,-1,1)
	return audio_feature
