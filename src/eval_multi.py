import os
import time
import einops
import sys
import cv2
import time
import numpy as np
import utils as ut
import config as cg
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from model.model_twostage import SlotAttentionAutoEncoder
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia
from kornia.augmentation.container import VideoSequential

def inference(masks_collection, gap, rgbs, gts, model, T, val_idx, n_f):
    masks = []
    scores = []
    for i in range(0, T, n_f):
        input = rgbs[:, i:i+n_f]
        bs = input.shape[1]
        if bs < n_f:
            bs = input.shape[1]
            input = torch.cat([input, rgbs[:, -(n_f-bs):]], dim=1)
        with torch.no_grad():
            semantic_mask, score, instance_mask, h, w = model(input)
        scores.append(score[0, :bs])
        masks.append(instance_mask[0, :bs])
    scores = torch.cat(scores, dim=0) # T K P
    scores = einops.rearrange(scores, 't k p -> t (k p)')
    masks = torch.cat(masks, dim=0) # T K P HW
    masks = einops.rearrange(masks, 't k p (h w) -> t (k p) h w', h=h)
    masks = masks.unsqueeze(2)
    for i in range(T):
        masks_collection[i].append(masks[i][scores[i]>=0.5].unsqueeze(0))
    return masks_collection

def eval(val_loader, model, device, moca, use_flow, it, gap, resultsPath=None, writer=None, train=False, n_f=7):
    with torch.no_grad():
        ious = {}
        ious_max = {}
        t = time.time()
        model.eval()
        mean = torch.tensor([0.43216, 0.394666, 0.37645])
        std = torch.tensor([0.22803, 0.22145, 0.216989])
        normalize_video = kornia.augmentation.Normalize(mean, std)
        aug_list = VideoSequential(
            normalize_video,
            data_format="BTCHW",
            same_on_frame=True)
        print(' --> running inference')
        if not os.path.exists('./DAVIS_Twostage'):
            os.makedirs('./DAVIS_Twostage')
        for idx, val_sample in enumerate(val_loader):           
            rgbs, gts, category, val_idx = val_sample
            rgbs = rgbs.float().to(device)  # b t c h w
            rgbs = aug_list(rgbs)
            gts = gts.float().to(device)  # b t c h w
            T = rgbs.shape[1]
            category = category[0]
            if category not in ious.keys(): 
                ious[category] = []
                ious_max[category] = []
            masks_collection = {}
            for i in range(T):
                masks_collection[i] = []
            masks_collection = inference(masks_collection, gap, rgbs, gts, model, T, val_idx, n_f)
            torch.save(masks_collection, 'DAVIS_Twostage/{}.pth'.format(category))

def main(args):
    lr = args.lr
    # args.lr = lr
    epsilon = 1e-5
    num_slots = args.num_slots
    num_instances = args.num_instances
    iters = args.num_iterations 
    batch_size = args.batch_size 
    warmup_it = int(args.warmup_steps)
    decay_step = int(args.decay_steps)
    num_it = int(args.num_train_steps)
    resume_path = args.resume_path
    attn_drop_f = args.attn_drop_f
    attn_drop_t = args.attn_drop_t
    path_drop = args.path_drop
    gap = args.gap
    num_frames = args.num_frames
    hid_dim = args.hidden_dim
    dino_path = args.dino_path
    out_channel = 3
    args.resolution = (480, 800)

    # initialize dataloader (validation bsz has to be 1 for FBMS, because of different resolutions, otherwise, can be >1)
    trn_dataset, val_dataset, resolution, in_out_channels, use_flow, loss_scale, ent_scale, cons_scale = cg.setup_dataset(args)
    trn_loader = ut.FastDataLoader(
        trn_dataset, num_workers=8, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = ut.FastDataLoader(
        val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('======> start inference {}, {}, use {}.'.format(args.dataset, args.verbose, device))

    model = SlotAttentionAutoEncoder(resolution=(256, 256),
                                num_slots=num_slots,
                                num_instances=num_instances,
                                in_channels=3, 
                                out_channels=out_channel,
                                hid_dim=hid_dim,
                                iters=iters,
                                path_drop=path_drop,
                                attn_drop_t=attn_drop_t,
                                attn_drop_f=attn_drop_f,
                                num_frames=num_frames,
                                dino_path=dino_path)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # initialize training

    it = 0
    if resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(resume_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        it = checkpoint['iteration']
        loss = checkpoint['loss']
        model.eval()
    else:
        print('no checkpouint found')
        sys.exit(0)

    if args.dataset == "MoCA": 
        moca = True
    else:
        moca = False

    eval(val_loader, model, device, moca, use_flow, it, gap=gap, resultsPath=None, train=False, n_f=num_frames)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    #optimization
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--num_train_steps', type=int, default=3e4)
    parser.add_argument('--warmup_steps', type=int, default=1e3)
    parser.add_argument('--decay_steps', type=int, default=1e4)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--loss_scale', type=float, default=100)
    parser.add_argument('--ent_scale', type=float, default=1.0)
    parser.add_argument('--cons_scale', type=float, default=1.0)
    parser.add_argument('--sudo_scale', type=float, default=0.0)
    parser.add_argument('--grad_iter', type=int, default=0)
    #settings
    parser.add_argument('--dataset', type=str, default='DAVIS', choices=['DAVIS', 'DAVIS2017', 'FBMS', 'STv2'])
    parser.add_argument('--with_rgb', action='store_true')
    parser.add_argument('--flow_to_rgb', action='store_true')
    parser.add_argument('--bi_cons', action='store_true')
    parser.add_argument('--entro_cons', action='store_true')
    parser.add_argument('--replicate', action='store_true')
    # architecture
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--num_slots', type=int, default=16)
    parser.add_argument('--num_instances', type=int, default=4)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--path_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop_f', type=float, default=0.0)
    parser.add_argument('--attn_drop_t', type=float, default=0.4)
    
    parser.add_argument('--gap', type=int, default=4, help='the sampling stride of frames')
    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--basepath', type=str, default="/home/ma-user/work/shuangrui/DAVIS-2016")
    parser.add_argument('--output_path', type=str, default="/home/ma-user/work/shuangrui/TEST_log/test")
    parser.add_argument('--dino_path', type=str, default="/home/ma-user/work/shuangrui/01_feature_warp/dino_deitsmall16_pretrain.pth")
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    args = parser.parse_args()
    args.inference = True
    args.flow_to_rgb = True
    main(args)
