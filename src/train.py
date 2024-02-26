import os
import time
import einops
import sys
import cv2
import numpy as np
import utils as ut
import config as cg
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment
from argparse import ArgumentParser
from model.model_twostage import SlotAttentionAutoEncoder
from model.unet import UNet
from utils import temporal_loss, warp_loss
import random
from utils import save_on_master, Augment_GPU_pre

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
    np.random.seed(seed)

    cudnn.benchmark = True
    print(f"GPU number:{torch.cuda.device_count()}")
    print(f"world_size:{ut.get_world_size()}")
    num_tasks = ut.get_world_size()
    lr = args.lr
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
    num_o = args.num_o
    num_t = args.num_t
    bi_cons = args.bi_cons
    entro_cons = args.entro_cons
    replicate = args.replicate
    gap = args.gap
    dino_path = args.dino_path
    num_frames = args.num_frames
    hid_dim = args.hidden_dim
    sudo_scale = args.sudo_scale
    grad_iter = args.grad_iter
    out_channel = 3
    args.resolution = (256, 256)
    aug_gpu = Augment_GPU_pre(args)
    
    # setup log and model path, initialize tensorboard,
    [logPath, modelPath, resultsPath] = cg.setup_path(args)
    print(logPath)

    # initialize dataloader (validation bsz has to be 1 for FBMS, because of different resolutions, otherwise, can be >1)
    trn_dataset, resolution, in_out_channels, use_flow, loss_scale, ent_scale, cons_scale = cg.setup_dataset(args)
    
    if True:  # args.distributed:
        num_tasks = ut.get_world_size()
        global_rank = ut.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            trn_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and logPath is not None:
        os.makedirs(logPath, exist_ok=True)
        writer = SummaryWriter(logPath)
    else:
        writer = None

    trn_loader = ut.FastDataLoader(
        trn_dataset, sampler=sampler_train, num_workers=8, batch_size=batch_size, 
        pin_memory=True, drop_last=True,
        multiprocessing_context="fork")
        
    model = SlotAttentionAutoEncoder(resolution=resolution,
                                     num_slots=num_slots,
                                     num_instances=num_instances,
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
                                     dino_path=dino_path)
    teacher = SlotAttentionAutoEncoder(resolution=resolution,
                                     num_slots=num_slots,
                                     num_instances=num_instances,
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
                                     teacher=True,
                                     dino_path=dino_path)

    teacher.load_state_dict(model.state_dict())
    teacher.to(device)
    teacher_without_ddp = teacher
    
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
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
                else:
                    encoder_reg.append(param)
            else:
                if name.endswith(".bias") or len(param.shape) == 1:
                    not_regularized.append(param)
                else:
                    regularized.append(param)
        return [{'params': encoder_reg}, {'params': encoder_noreg, 'weight_decay': 0.},
            {'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
    
    param_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    optimizer.param_groups[0]['weight_decay'] = 0.04
    criterion = nn.CrossEntropyLoss()

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

    lr_scheduler = cosine_scheduler(lr, 0.1*lr, num_it//len(trn_loader), len(trn_loader), 0)
    wd_scheduler = cosine_scheduler(0.1, 0.1, num_it//len(trn_loader), len(trn_loader))
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    for p in teacher.parameters():
        p.requires_grad = False

    #save every save_freq iterations
    log_freq = 100 #training log
    save_freq = 5e2

    print('======> start training {}, {}, use {}.'.format(args.dataset, args.verbose, device))
    timestart = time.time()
    iter_per_epoch = int(10000 // (num_tasks * args.batch_size))
    
    # overfit single batch for debug
    sample = next(iter(trn_loader))
    while it < num_it:
        if args.distributed:
            trn_loader.sampler.set_epoch(it//iter_per_epoch)
        for _, sample in enumerate(trn_loader):
            for i, param_group in enumerate(optimizer.param_groups):
                if i < 2:
                    weight = 0.0 if it < grad_iter else 0.01
                else:
                    weight = 1.0
                param_group["lr"] = lr_scheduler[it] * weight
                if i % 2 == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_scheduler[it]

            model.train()
            teacher.eval()
            optimizer.zero_grad()
           
            rgb, flow_idxs = sample
            rgb = rgb.float().to(device)
            flow_idxs = flow_idxs.to(device)
            
            sample_weight = 1.0 - it / num_it * 0.99
            motion_mask, _, semantic, instance, objects = model(aug_gpu(rgb), sample_weight, flow_idxs)
            motion_mask_t, feat_t, semantic_t, instance_t, objects_t = teacher(aug_gpu(rgb), sample_weight, flow_idxs)
            
            differ_loss = ent_scale * torch.einsum('btmc,btnc->btmn', 
                F.normalize(motion_mask.softmax(dim=-2)+1e-8, dim=-1), F.normalize(motion_mask.softmax(dim=-2)+1e-8, dim=-1))
            mask = torch.ones_like(differ_loss)
            mask[:, :, torch.arange(num_slots), torch.arange(num_slots)] = 0
            differ_loss = torch.sum(differ_loss*mask) / torch.sum(mask) if entro_cons else 0

            temporal_diff = dense_transport_loss(motion_mask, motion_mask_t, feat_t, feat_t)
            consistency_loss = cons_scale * temporal_diff if bi_cons else 0

            valid_instance = filter_valid_instance(motion_mask, semantic, instance)
            valid_instance_t = filter_valid_instance(motion_mask_t, semantic_t, instance_t)
            instance_loss = object_loss(valid_instance, valid_instance_t, objects, objects_t)
            loss = consistency_loss + differ_loss + instance_loss
            loss.backward()
            optimizer.step()

            if it % log_freq == 0 and writer is not None:
                print('iteration {},'.format(it),
                  'time {:.01f}s,'.format(time.time() - timestart),
                  'learning rate {:.05f}'.format(lr_scheduler[it]),
                  'total loss {:.05f}'.format(loss.detach().cpu().numpy()),
                  'differ loss {:.05f}.'.format(float(differ_loss)),
                  'instance loss {:.05f}.'.format(float(instance_loss)),
                  'bidirection consistency loss {:.010f}.'.format(float(consistency_loss)))
                timestart = time.time()
            # save model
            if it % save_freq == 0 and it > 0:
                filename = os.path.join(modelPath, 'checkpoint_{}_loss_{}.pth'.format(it, np.round(loss.item(), 3)))
                save_on_master({
                    'iteration': it,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, filename)
            with torch.no_grad():
                momentum = 0.0 if it < 0 else 0.999
                for (name_q, param_q), (name_k, param_k) in zip(model.named_parameters(), teacher.named_parameters()):
                    param_k.data = param_k.data * momentum + param_q.data * (1-momentum)

            it += 1


def dense_transport_loss(map_s, map_t, token_s, token_t):
    # map b t s hw
    # cost b hw hw
    # token b t hw c
    cost_matrix = []
    mu_s = []
    mu_t = []
    cross_matrix = []
    token_s = F.normalize(token_s, dim=-1)
    token_t = F.normalize(token_t, dim=-1)
    for i in range(map_s.shape[1]):
        for j in range(map_t.shape[1]):
            if i == j:
                continue
            cost = torch.einsum('bnc,bmc->bnm', token_s[:, i], token_t[:, j])
            cost_matrix.append(cost)
            mu_s.append(torch.abs(token_s[:, i]).sum(-1))
            mu_t.append(torch.abs(token_t[:, j]).sum(-1))
            cross = torch.sum(F.softmax(map_t[:, j].unsqueeze(-2), dim=1)*torch.log(F.softmax(map_s[:, i].unsqueeze(-1), dim=1)+1e-8), dim=1)
            cross_matrix.append(cross)
    cost_matrix = torch.cat(cost_matrix, dim=0)
    mu_s = torch.cat(mu_s, dim=0)
    mu_t = torch.cat(mu_t, dim=0)
    cross_matrix = torch.cat(cross_matrix, dim=0)
    # mu_s = torch.ones(map_s.shape[-1]).to(map_s.device) / map_s.shape[-1]
    # mu_t = torch.ones(map_t.shape[-1]).to(map_t.device) / map_t.shape[-1]
    # with torch.no_grad():
    #     T = sinkhorn_cluster(mu_s.unsqueeze(0), mu_t.unsqueeze(0), 1-cost_matrix)
    with torch.no_grad():
        T = sinkhorn_cluster(mu_s, mu_t, 1-cost_matrix)
    loss = - cross_matrix * T
    loss = torch.mean(loss, 0).sum()
    return loss


def sinkhorn_cluster(mu_s, mu_t, M, eps=0.05):
    K = torch.exp(-M/eps)
    mu_s = mu_s / torch.sum(mu_s, dim=-1, keepdim=True)
    mu_t = mu_t / torch.sum(mu_t, dim=-1, keepdim=True)
    b = torch.ones_like(mu_t)
    for i in range(3):
        a = mu_s / torch.sum(K*b[:,None,:], -1)
        b = mu_t / torch.sum(K*a[:,:,None], 1)
    T = a[:,:,None] * K * b[:,None,:]
    return T


def filter_valid_instance(mask, semantic, instance):
    # mask b t s hw
    # semantic b t s c
    # instance b t s p c
    # valid b t s p
    mask = F.softmax(mask, dim=-2)
    mask = (mask>0.3).float()
    mask = torch.mean(mask, dim=-1, keepdim=True)
    semantic = F.normalize(semantic, dim=-1)
    instance = F.normalize(instance, dim=-1)
    similarity = torch.einsum('btsc,btspc->btsp', semantic, instance)
    valid_sample = (similarity>0.5).float() * (mask>0.2).float()
    return valid_sample


def matching_cost(similarity):
    cost = (1 - similarity).cpu()
    b, s, p, q = cost.shape
    indices = []
    for i in range(b):
        for j in range(s):
            row, col = linear_sum_assignment(cost[i, j])
            row = torch.LongTensor(row)
            col = torch.LongTensor(col)
            indice = torch.arange(q)
            indice = indice.unsqueeze(0).repeat([p, 1])
            indice[row, col] = 0
            indice[row, 0] = col
            indices.append(indice)
    indices = torch.stack(indices, 0).view(b, s, p, q).to(similarity.device)
    return indices


def object_loss(valid_instance, valid_instance_t, objects, objects_t):
    # valid b t s p
    # object b t s p c
    loss_stat = []
    valid_stat = []
    for i in range(objects.shape[1]):
        for j in range(objects_t.shape[1]):
            instance = objects[:, i]
            instance_t = objects_t[:, j]
            similarity = torch.einsum('bspc,bsqc->bspq', instance, instance_t)
            with torch.no_grad():
                sort_idx = matching_cost(similarity)
            # sort_idx = torch.argsort(similarity, dim=-1, descending=True) # b s p q
            valid_t = torch.gather(valid_instance_t[:, j], dim=-1, index=sort_idx[:, :, :, 0]) # b s p
            sort_idx = (sort_idx.unsqueeze(-1)).repeat(1, 1, 1, 1, objects.shape[-1])
            pos = torch.gather(instance_t, dim=-2, index=sort_idx[:, :, :, 0]) # b s p c
            repeat_instance_t = (instance_t.unsqueeze(2)).repeat(1, 1, objects.shape[-2], 1, 1)
            neg = torch.gather(repeat_instance_t, dim=-2, index=sort_idx[:, :, :, 1:]) # b s p q-1 c
            loss = valid_instance[:, i] * (F.relu(torch.sum(instance.unsqueeze(-2)*neg, dim=-1)-0.5).mean(dim=-1) \
                - valid_t * torch.sum(instance*pos, dim=-1))
            loss_stat.append(loss)
            valid_stat.append(valid_instance[:, i])
    loss_stat = torch.cat(loss_stat, dim=0)
    valid_stat = torch.cat(valid_stat, dim=0)
    loss = torch.sum(loss_stat) / torch.sum(valid_stat + 1e-10)
    return loss


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
    print(args.replicate)
    args.inference = False
    args.distributed = True
    main(args)
