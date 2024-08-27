import os
import time
import einops
import sys
import cv2
import numpy as np
import utils as ut
import config as cg
import clip
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
import kornia
from kornia.augmentation import VideoSequential
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from model.model_action import SlotAttentionAutoEncoder
import random
from utils import save_on_master
from data import ActionData
from progress.bar import Bar

def Augment_GPU_Train(args):
    # crop_size = args.crop_size
    resolution = args.resolution
    radius_0 = int(0.1*resolution[0])//2*2 + 1
    radius_1 = int(0.1*resolution[1])//2*2 + 1
    sigma = random.uniform(0.1, 2)
    # For k400 parameter:
    # mean = torch.tensor([0.43216, 0.394666, 0.37645])
    # std = torch.tensor([0.22803, 0.22145, 0.216989])
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    normalize_video = kornia.augmentation.Normalize(mean, std)
    aug_list = VideoSequential(
        kornia.augmentation.RandomResizedCrop(size=resolution, scale=(0.8, 1.0)),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        kornia.augmentation.RandomGaussianBlur((radius_0, radius_1), (sigma, sigma), p=0.5),
        normalize_video,
        data_format="BTCHW",
        same_on_frame=True)
    return aug_list

def Augment_GPU_Val(args):
    resolution = args.resolution
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    normalize_video = kornia.augmentation.Normalize(mean, std)
    aug_list = VideoSequential(
        normalize_video,
        data_format="BTCHW",
        same_on_frame=True)
    return aug_list

def all_gather(tensor, expand_dim=0, num_replicas=8):
    """Gathers a tensor from other replicas, concat on expand_dim and return"""
    num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
    other_replica_tensors = [torch.zeros_like(tensor) for _ in range(num_replicas)]
    dist.all_gather(other_replica_tensors, tensor)
    other_replica_tensors[dist.get_rank()] = tensor
    return torch.cat([o for o in other_replica_tensors], expand_dim)

def main(args):
    ut.init_distributed_mode(args)
    torch.autograd.set_detect_anomaly(True)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + ut.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    print(f"GPU number:{torch.cuda.device_count()}")
    print(f"world_size:{ut.get_world_size()}")
    num_tasks = ut.get_world_size()
    lr = args.lr
    num_slots = args.num_slots
    iters = args.num_iterations
    batch_size = args.batch_size
    num_it = int(args.num_train_steps)
    attn_drop_f = args.attn_drop_f
    attn_drop_t = args.attn_drop_t
    path_drop = args.path_drop
    num_o = args.num_o
    num_t = args.num_t
    entro_cons = args.entro_cons
    dino_path = args.dino_path
    num_frames = args.num_frames
    hid_dim = args.hidden_dim
    grad_iter = args.grad_iter
    out_channel = 3
    args.resolution = (224, 224)
    aug_gpu = Augment_GPU_Train(args)
    aug_val = Augment_GPU_Val(args)

    # setup log and model path, initialize tensorboard,
    [logPath, modelPath, resultsPath] = cg.setup_path(args)
    print(logPath)

    resolution = args.resolution
    trn_dataset = ActionData(data_dir='/path/to/dataset', resolution=args.resolution, seq_length=num_frames, train=True)
    val_dataset = ActionData(data_dir='/path/to/dataset', resolution=args.resolution, seq_length=num_frames, train=False)
    
    if True:  # args.distributed:
        num_tasks = ut.get_world_size()
        global_rank = ut.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            trn_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(trn_dataset)
        sampler_val = torch.utils.data.RandomSampler(val_dataset)

    if global_rank == 0 and logPath is not None:
        os.makedirs(logPath, exist_ok=True)
        writer = SummaryWriter(logPath)
    else:
        writer = None

    trn_loader = ut.FastDataLoader(
        trn_dataset, sampler=sampler_train, num_workers=8, batch_size=batch_size, 
        pin_memory=True, drop_last=True,
        multiprocessing_context="fork")
    val_loader = ut.FastDataLoader(
        val_dataset, sampler=sampler_val, num_workers=8, batch_size=batch_size,
        pin_memory=True, drop_last=False,
        multiprocessing_context="fork")
    
    model = SlotAttentionAutoEncoder(resolution=resolution,
                                     num_slots=num_slots,
                                     in_channels=3, 
                                     out_channels=out_channel,
                                     hid_dim=hid_dim,
                                     iters=iters,
                                     path_drop=path_drop,
                                     attn_drop_t=attn_drop_t,
                                     attn_drop_f=attn_drop_f,
                                     num_o=num_o,
                                     num_t=num_t,
                                     num_frames=num_frames,
                                     teacher=False,
                                     slot_tune=False,
                                     background=args.background,
                                     task='joint',
                                     student_model=args.student,
                                     teacher_model=args.teacher,
                                     correlation=args.correlation,
                                     dino_path=dino_path)
    
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    teacher, _ = clip.load('ViT-B/32')
    teacher.to(device)

    it = 0

    def get_params_groups(model):
        encoder_reg = []
        encoder_noreg = []
        regularized = []
        not_regularized = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if 'encoder' in name:
                if name.endswith(".bias") or len(param.shape) == 1:
                    encoder_noreg.append(param)
                elif 'slots_embed' in name:
                    encoder_noreg.append(param)
                else:
                    encoder_reg.append(param)
            else:
                if name.endswith(".bias") or len(param.shape) == 1:
                    not_regularized.append(param)
                elif 'position_embed' in name or 'encoder_pos' in name:
                    not_regularized.append(param)
                elif 'time_embed' in name or 'slots_embed' in name:
                    not_regularized.append(param)
                elif 'st_token' in name:
                    not_regularized.append(param)
                else:
                    regularized.append(param)
        return [{'params': encoder_reg}, {'params': encoder_noreg, 'weight_decay': 0.},
            {'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
    
    param_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    optimizer.param_groups[0]['weight_decay'] = 0.04

    def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule
    lr_scheduler = cosine_scheduler(lr, 1e-5, num_it//len(trn_loader), len(trn_loader), 0)
    wd_scheduler = cosine_scheduler(0.1, 0.1, num_it//len(trn_loader), len(trn_loader))
        
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
    for p in teacher.parameters():
        p.requires_grad = False
    for p in model.module.encoder.parameters():
        p.requires_grad = False

    log_freq = 100 #report train iou to tensorboard

    print('======> start training {}, {}, use {}.'.format(args.dataset, args.verbose, device))
    timestart = time.time()
    iter_per_epoch = int(len(trn_dataset) // (num_tasks * args.batch_size))

    while it < num_it:

        if args.distributed:
            trn_loader.sampler.set_epoch(it//iter_per_epoch)
            val_loader.sampler.set_epoch(it//iter_per_epoch)
        
        for _, sample in enumerate(trn_loader):
            for i, param_group in enumerate(optimizer.param_groups):
                if i < 2:
                    weight = 0.0 if it < grad_iter else 1.0
                else:
                    weight = 1.0
                param_group["lr"] = lr_scheduler[it] * weight
                if i % 2 == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_scheduler[it]

            model.eval()
            teacher.eval()
            model.module.st_block.train()
            optimizer.zero_grad()
           
            #'rgb' has shape B, 5, C, H, W 
            #'gt' has shape B, 3, C, H, W 
            rgb, label = sample
            rgb = rgb.float().to(device)
            label = label.long().to(device)
            feats, motion_mask, slot_warp, st_token = model(aug_gpu(rgb), 0, 0)
            logit, st_token = st_token

            with torch.no_grad():
                input_tensor = einops.rearrange(aug_val(rgb), 'b t c h w -> (b t) c h w')
                cls_t = teacher.module.encode_image(input_tensor).float()
                cls_t = einops.rearrange(cls_t, '(b t) c -> b t c', t=num_frames)

            differ_loss = torch.einsum('btmc,btnc->btmn', 
                F.normalize(motion_mask.softmax(dim=-2)+1e-8, dim=-1), F.normalize(motion_mask.softmax(dim=-2)+1e-8, dim=-1))
            mask = torch.ones_like(differ_loss)
            mask[:, :, torch.arange(num_slots), torch.arange(num_slots)] = 0
            differ_loss = torch.sum(differ_loss*mask) / torch.sum(mask) if entro_cons else 0

            st_token_all = all_gather(st_token, num_replicas=None)
            label_all = all_gather(label, num_replicas=None)
            st_token_all = F.normalize(st_token_all, dim=-1, p=2)
            st_token = F.normalize(st_token, dim=-1, p=2)
            positive_index = (label.unsqueeze(-1) == label_all.unsqueeze(0))
            negative_index = (label.unsqueeze(-1) != label_all.unsqueeze(0))
            query = st_token
            key = st_token_all.detach()
            random_order = torch.randperm(query.shape[1])
            distance = torch.einsum('btsc,ntsc->bnts', query, key[:, random_order])
            object_distance = torch.einsum('btsc,btnc->btsn', query, query.detach())
            object_index = torch.zeros_like(object_distance)
            object_index[:, :, torch.arange(query.shape[-2]), torch.arange(query.shape[-2])] = 1
            object_index = (object_index == 0)
            positive_distance = distance[positive_index].mean()
            negative_distance = F.relu(distance[negative_index]-0.3).mean() + F.relu(object_distance[object_index]-0.3).mean()
            time_loss = negative_distance - positive_distance

            cls_t = F.normalize(cls_t, dim=-1)
            adj = torch.einsum('btsc,ptc->bpts', slot_warp, cls_t)
            adj = adj * torch.softmax(adj/0.1, dim=-1)
            adj = torch.sum(adj, dim=-1)
            mask = torch.zeros_like(adj)
            mask[torch.arange(adj.shape[0]), torch.arange(adj.shape[1])] = 1
            s2c = - torch.log(torch.sum(torch.exp(adj/0.07)*mask, dim=1)/torch.sum(torch.exp(adj/0.07), dim=1))
            c2s = - torch.log(torch.sum(torch.exp(adj/0.07)*mask, dim=0)/torch.sum(torch.exp(adj/0.07), dim=0))
            contrast_loss = s2c.mean() + c2s.mean()

            acc = torch.sum(torch.argmax(logit, dim=1)==label)
            acc = acc / logit.shape[0]
            logit_loss = F.cross_entropy(logit, label)
            loss = 0.05 * differ_loss + 0.2 * time_loss + 0.2 * contrast_loss + logit_loss
            loss.backward()

            optimizer.step()

            if it % log_freq == 0 and writer is not None:
                print('iteration {},'.format(it),
                  'time {:.01f}s,'.format(time.time() - timestart),
                  'learning rate {:.05f}'.format(lr_scheduler[it]),
                  'classification loss {:.05f}'.format(logit_loss.detach().cpu().numpy()),
                  'contrast loss {:.05f}'.format(contrast_loss.detach().cpu().numpy()),
                  'differ loss {:.05f}'.format(differ_loss.detach().cpu().numpy()),
                  'time loss {:.05f}'.format(time_loss.detach().cpu().numpy()),
                  'acc {:.03f}'.format(acc.detach().cpu().numpy()))
                timestart = time.time()
            it += 1

        with torch.no_grad():
            bar = Bar('Processing', max=len(val_loader))
            accs = 0
            for batch_idx, sample in enumerate(val_loader):
                model.eval()
                rgb, label = sample
                rgb = rgb.float().to(device)
                label = label.long().to(device)
                _, _, _, st_token = model(aug_val(rgb), 0, 0)
                st_token = st_token[0]
                acc = torch.sum(torch.argmax(st_token, dim=1)==label, dim=0, keepdim=True).float()
                acc = all_gather(acc, num_replicas=None)
                accs += torch.sum(acc)
                bar.suffix = '({batch}/{size}) Acc: {acc:.3f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    acc=torch.sum(torch.argmax(st_token, dim=1)==label).float().item()/st_token.shape[0]
                )
                bar.next()
            bar.finish()
            accs = accs / len(val_dataset)

        filename = os.path.join(modelPath, 'checkpoint_{}_acc_{}.pth'.format(it, np.round(accs.cpu().numpy(), 3)))
        save_on_master({
            'iteration': it,
            'model_state_dict': model_without_ddp.state_dict(),
            'acc': accs,
            }, filename)


if __name__ == "__main__":
    parser = ArgumentParser()
    #optimization
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--num_train_steps', type=int, default=1e4) #300k
    parser.add_argument('--warmup_steps', type=int, default=1e3)
    parser.add_argument('--decay_steps', type=int, default=1e4)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--loss_scale', type=float, default=100)
    parser.add_argument('--ent_scale', type=float, default=1.0)
    parser.add_argument('--cons_scale', type=float, default=1.0)
    parser.add_argument('--sudo_scale', type=float, default=0.0)
    parser.add_argument('--grad_iter', type=int, default=0)
    #settings
    parser.add_argument('--dataset', type=str, default='DAVIS', choices=['DAVIS', 'MoCA', 'FBMS', 'STv2'])
    parser.add_argument('--with_rgb', action='store_true')
    parser.add_argument('--flow_to_rgb', action='store_true')
    parser.add_argument('--entro_cons', action='store_true')
    parser.add_argument('--replicate', action='store_true')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--correlation', type=str, default='none')
    # architecture
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--num_slots', type=int, default=8)
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--path_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop_f', type=float, default=0.0)
    parser.add_argument('--attn_drop_t', type=float, default=0.4)
    parser.add_argument('--num_o', type=int, default=1)
    parser.add_argument('--num_t', type=int, default=1)
    parser.add_argument('--student', type=str, default='dino')
    parser.add_argument('--teacher', type=str, default='dino')
    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--verbose', type=str, default=None)
    parser.add_argument('--dino_path', type=str, default="/path/to/dino")
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    
    args = parser.parse_args()
    print(args.replicate)
    args.inference = False
    args.distributed = True
    main(args)
