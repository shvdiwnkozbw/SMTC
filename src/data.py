import os
import cv2
import glob
import torch
import random
import einops
import numpy as np
import glob as gb
from utils import read_flo
from torch.utils.data import Dataset


def readRGB(sample_dir, resolution):
    rgb = cv2.imread(sample_dir)
    try:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    except:
        print(sample_dir)
    rgb = rgb / 255
    rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    return einops.rearrange(rgb, 'h w c -> c h w')

def readSeg(sample_dir, resolution=None):
    gt = cv2.imread(sample_dir) / 255
    if resolution:
        gt = cv2.resize(gt, (resolution[1], resolution[0]), interpolation=cv2.INTER_NEAREST)
    return einops.rearrange(gt, 'h w c -> c h w')

    
class Dataloader(Dataset):
    def __init__(self, data_dir, resolution, dataset, seq_length=7, gap=2, to_rgb=False, train=True, val_seq=None):
        self.dataset = dataset
        self.eval = eval
        self.to_rgb = to_rgb
        self.data_dir = data_dir
        self.img_dir = data_dir[1]
        self.gap = gap
        self.resolution = resolution
        self.seq_length = seq_length
        if train:
            self.train = train
            self.seq = list([os.path.basename(x) for x in gb.glob(os.path.join(self.img_dir, '*'))])
        else: 
            self.train = train
            self.seq = val_seq
        

    def __len__(self):
        if self.train:
            return 10000
        else:
            return len(self.seq)

    def __getitem__(self, idx):
        if self.train:
            seq_name = random.choice(self.seq)
            seq = os.path.join(self.img_dir, seq_name, '*.jpg')
            imgs = gb.glob(seq)
            imgs.sort()
            length = len(imgs)
            gap = self.gap
            if gap*self.seq_length//2 >= length-gap*self.seq_length//2-1:
                gap = gap-1
            ind = random.randint(gap*self.seq_length//2, length-gap*self.seq_length//2-1)
            
            seq_ids = [ind+gap*(i-self.seq_length//2) for i in range(self.seq_length)]

            flow_idxs = []
            for i in range(self.seq_length):
                if i == 0:
                    flow_idxs.extend(np.random.choice(np.arange(1, self.seq_length), 2, replace=False).tolist())
                elif i == self.seq_length-1:
                    flow_idxs.extend(np.random.choice(np.arange(self.seq_length-1), 2, replace=False).tolist())
                else:
                    flow_idxs.extend([np.random.choice(i), np.random.choice(np.arange(i+1, self.seq_length))])

            rgb_dirs = [imgs[i] for i in seq_ids]
            flow_idxs = np.array(flow_idxs)
            rgbs = [readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs]
            out_rgb = np.stack(rgbs, 0)
            
            return out_rgb, flow_idxs
        else:
            if self.dataset == 'FBMS':
                seq_name = self.seq[idx]
                rgb_dirs = sorted(os.listdir(os.path.join(self.data_dir[1], seq_name)))
                rgb_dirs = [os.path.join(self.data_dir[1], seq_name, x) for x in rgb_dirs if x.endswith(".jpg")]
                
                rgbs = np.stack([readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs], axis=0)
                gt_dirs = os.listdir(os.path.join(self.data_dir[2], seq_name))
                gt_dirs = sorted([gt for gt in gt_dirs if gt.endswith(".png")])
                val_idx = [int(x[:-4])-int(gt_dirs[0][:-4]) for x in gt_dirs if x.endswith(".png")]
                gt_dirs = [os.path.join(self.data_dir[2], seq_name, x) for x in gt_dirs if x.endswith(".png")]  
                gts = np.stack([readSeg(gt_dir) for gt_dir in gt_dirs], axis=0)
                return rgbs, gts, seq_name, val_idx
            else:
                seq_name = self.seq[idx]
                tot = len(glob.glob(os.path.join(self.data_dir[1], seq_name, '*')))
                rgb_dirs = [os.path.join(self.data_dir[1], seq_name, str(i).zfill(5)+'.jpg') for i in range(tot)]
                gt_dirs = [os.path.join(self.data_dir[2], seq_name, str(i).zfill(5)+'.png') for i in range(tot)]
                rgbs = np.stack([readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs], axis=0)
                gts = np.stack([readSeg(gt_dir) for gt_dir in gt_dirs], axis=0)
                return rgbs, gts, seq_name, [i for i in range(tot)]
                

class ActionData(Dataset):
    def __init__(self, data_dir, resolution, seq_length=8, train=True):
        self.eval = eval
        self.img_dir = os.path.join(data_dir, '20bn-something-something-v2-frames')
        self.resolution = resolution
        self.seq_length = seq_length
        self.train = train
        if train:
            video_folder = os.path.join(data_dir, 'labels', 'train_videofolder.txt')
        else:
            video_folder = os.path.join(data_dir, 'labels', 'val_videofolder.txt')
        self.seq = {}
        with open(video_folder, 'r') as f:
            while True:
                row = f.readline()
                if not len(row):
                    break
                vid, frame, label = row.split(' ')
                label = int(label[:-1]) if label.endswith('\n') else int(label)
                self.seq[vid] = label
        
    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if self.train:
            seq_name = random.choice(list(self.seq.keys()))
        else:
            keys = list(self.seq.keys())
            keys.sort()
            seq_name = keys[idx]
        label = self.seq[seq_name]
        seq = os.path.join(self.img_dir, seq_name, '*.jpg')
        imgs = gb.glob(seq)
        imgs.sort()
        length = len(imgs)
        gap = length // self.seq_length
        ind = random.randint(0, length-(self.seq_length-1)*gap-1)
        
        seq_ids = [ind+gap*i for i in range(self.seq_length)]

        rgb_dirs = [imgs[i] for i in seq_ids]
        rgbs = [readRGB(rgb_dir, self.resolution) for rgb_dir in rgb_dirs]
        out_rgb = np.stack(rgbs, 0) ## T, C, H, W 
        
        return out_rgb, label
