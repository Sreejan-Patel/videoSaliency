import sys
import os
import numpy as np
import cv2
import torch
from avinet_model import AViNetModel
from scipy.ndimage.filters import gaussian_filter
from loss_util import *
import argparse

from torch.utils.data import DataLoader
from dataloader import DHF1KDataset
from utils import *
import time
from tqdm import tqdm
from torchvision import transforms, utils
from os.path import join
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='DIEM', type=str, help='dataset name')
parser.add_argument('--weights', default='avinet_model.pt', type=str, help='path to avinet weights file')
parser.add_argument('--input_path', default='result', type=str, help='path for input files')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def evaluate():
    args = parser.parse_args()
	# optional two command-line arguments
    path_indata = args.input_path
    file_weight = args.file_weight

    len_temporal = args.clip_size

    model = AViNetModel()

    model.load_state_dict(torch.load(file_weight))

    model = model.to(device)
    torch.backends.cudnn.benchmark = False
    model.eval()

    # iterate over the path_indata directory
    list_indata = []
    with open('./DIEM_list_test_fps.txt', 'r') as f:
        for line in f.readlines():
            name = line.split(' ')[0].strip()
            list_indata.append(name)

    list_indata.sort()

    frame_sim_loss = 0
    frame_cc_loss = 0
    frame_nss_loss = 0
    frame_aucj_loss = 0
    frame_cnt = 0

    avg_video_sim_loss = 0
    avg_video_cc_loss = 0
    avg_video_nss_loss = 0
    avg_video_aucj_loss = 0
    num_videos = 0

    for dname in list_indata:
        print("="*25)
        print ('processing ' + dname, flush=True)
        list_frames = [f for f in os.listdir(os.path.join(path_indata, 'video_frames', 'DIEM', dname)) if os.path.isfile(os.path.join(path_indata, 'video_frames', 'DIEM', dname, f))]
        list_frames.sort()

        video_sim_loss = 0
        video_cc_loss = 0
        video_nss_loss = 0
        video_aucj_loss = 0
        num_frames = 0

        if len(list_frames) >= 2*len_temporal-1:

            snippet = []
            for i in range(len(list_frames)):
                torch_img, img_size = torch_transform(os.path.join(path_indata, 'video_frames', 'DIEM', dname, list_frames[i]))

                snippet.append(torch_img)
                
                if i >= len_temporal-1:
                    clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                    clip = clip.permute((0,2,1,3,4))

                    sim_loss, cc_loss, nss_loss, aucj_loss = process(model, clip, path_indata, dname, list_frames[i], args, img_size)
                    # print(cc_loss)
                    if np.isnan(sim_loss) or np.isnan(cc_loss) or np.isnan(nss_loss):
                        print("1", dname, list_frames[i])
                        print("No saliency")
                    else:
                        frame_sim_loss += sim_loss
                        frame_nss_loss += nss_loss
                        frame_cc_loss += cc_loss
                        frame_aucj_loss += aucj_loss
                        frame_cnt += 1

                        video_sim_loss += sim_loss
                        video_nss_loss += nss_loss
                        video_cc_loss += cc_loss
                        video_aucj_loss += aucj_loss
                        num_frames += 1
                    # process first (len_temporal-1) frames
                    if i < 2*len_temporal-2:
                        sim_loss, cc_loss, nss_loss, aucj_loss = process(model, torch.flip(clip, [2]), path_indata, dname, list_frames[i-len_temporal+1], args, img_size)
                        if np.isnan(sim_loss) or np.isnan(cc_loss) or np.isnan(nss_loss):
                            print("2", dname, list_frames[i])
                            print("No saliency")
                        else:
                            frame_sim_loss += sim_loss
                            frame_nss_loss += nss_loss
                            frame_cc_loss += cc_loss
                            frame_aucj_loss += aucj_loss
                            frame_cnt += 1

                            video_sim_loss += sim_loss
                            video_nss_loss += nss_loss
                            video_cc_loss += cc_loss
                            video_aucj_loss += aucj_loss
                            num_frames += 1

                    del snippet[0]
                    # print(frame_cnt, frame_sim_loss)

        else:
            print (' more frames are needed')
            print("non weighted")

        num_videos += 1
        avg_video_sim_loss += video_sim_loss / num_frames
        avg_video_nss_loss += video_nss_loss / num_frames
        avg_video_cc_loss += video_cc_loss / num_frames
        avg_video_aucj_loss += video_aucj_loss / num_frames

    print("SIM:", frame_sim_loss/frame_cnt)
    print("CC:", frame_cc_loss/frame_cnt)
    print("NSS:", frame_nss_loss/frame_cnt)
    print("AUCJ:", frame_aucj_loss/frame_cnt)


    print("Avg Video SIM:", avg_video_sim_loss/num_videos)
    print("Avg Video CC:", avg_video_cc_loss/num_videos)
    print("Avg Video NSS:", avg_video_nss_loss/num_videos)
    print("Avg Video AUCJ:", avg_video_aucj_loss/num_videos)

		
def torch_transform(path):
	img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
	])
	img = Image.open(path).convert('RGB')
	sz = img.size
	img = img_transform(img)
	return img, sz

def blur(img):
	k_size = 11
	bl = cv2.GaussianBlur(img,(k_size,k_size),0)
	return torch.FloatTensor(bl)

def get_fixation(path_indata, dname, _id):
	info = sio.loadmat(join(path_indata, 'annotations/DIEM', dname, 'fixMap_{}.mat'.format(_id)))
	return info['eyeMap']

def process(model, clip, path_indata, dname, frame_no, args, img_size):
	with torch.no_grad():
		smap = model(clip.to(device)).cpu().data[0]
	
	smap = smap.numpy()
	_id = frame_no.split('.')[0].split('_')[-1]
	gt = cv2.imread(join(path_indata, 'annotations/DIEM', dname, 'maps', 'eyeMap_{}.jpg'.format(_id)), 0)
	smap = cv2.resize(smap, (gt.shape[1], gt.shape[0]))
	fix = get_fixation(path_indata, dname, _id)
	smap = blur(smap)
	
	gt = torch.FloatTensor(gt).unsqueeze(0)
	fix = torch.FloatTensor(fix).unsqueeze(0)
	smap = smap.unsqueeze(0)
	# print(smap.size(), gt.size())
	sim_loss = similarity(smap, gt)
	cc_loss = cc(smap, gt)
	nss_loss = nss(smap, fix)
	aucj_loss = auc_judd(smap, fix)

	if np.isnan(sim_loss) or np.isnan(cc_loss) or np.isnan(nss_loss):
		assert gt.numpy().max()==0, gt.numpy().max()
	return sim_loss, cc_loss, nss_loss, aucj_loss

	
if __name__ == '__main__':
	evaluate()

