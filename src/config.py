import os
import torch
import itertools
import glob as gb
import numpy as np
import json
from datetime import datetime
from data import Dataloader


def setup_path(args):
    dataset = args.dataset
    num_slots = args.num_slots
    iters = args.num_iterations
    batch_size = args.batch_size
    resolution = args.resolution
    flow_to_rgb = args.flow_to_rgb
    gap = args.gap
    dim = args.hidden_dim
    verbose = args.verbose if args.verbose else 'none'
    flow_to_rgb_text = 'rgb' if flow_to_rgb else 'uv'
    inference = args.inference

    # make all the essential folders, e.g. models, logs, results, etc.
    global dt_string, logPath, modelPath, resultsPath
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    dir_name = f'{dt_string}-{dataset}-{flow_to_rgb_text}-dim_{dim}' + \
                   f'gap_{gap}_-lr_{args.lr}-bs_{batch_size}'

    logPath = os.path.join(args.output_path, dir_name, 'log')

    modelPath = os.path.join(args.output_path, dir_name, 'model')

    if inference:
        resultsPath = os.path.join(args.output_path, dir_name, 'results', args.resume_path.split('/')[-1])
        os.makedirs(resultsPath, exist_ok=True)
    else:
        os.makedirs(logPath, exist_ok=True)
        os.makedirs(modelPath, exist_ok=True)
        resultsPath = logPath

        # save all the expersetup_datasetiment settings.
        if args.rank == 0:
            with open('{}/running_command.txt'.format(modelPath), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    return [logPath, modelPath, resultsPath]


def setup_dataset(args):
    resolution = args.resolution  # h,w
    res = ""
    with_gt = True
    pairs = [1, 2, -1, -2]
    if args.dataset == 'DAVIS':
        basepath = args.basepath
        img_dir = basepath + '/JPEGImages/480p'
        gt_dir = basepath + '/Annotations/480p'


        val_flow_dir = basepath + '/Flows_gap1/'
        val_seq = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees', 
                    'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane', 
                    'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch']       
        val_data_dir = [val_flow_dir, img_dir, gt_dir]
        res = "480p"

    elif args.dataset == 'DAVIS2017':
        basepath = args.basepath
        img_dir = basepath + '/JPEGImages/480p'
        gt_dir = basepath + '/Annotations/480p'

        val_flow_dir = basepath + '/Flows_gap1/'
        val_seq = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees', 
                    'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane', 
                    'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch',
                    'bike-packing', 'dogs-jump', 'gold-fish', 'india', 'judo', 'lab-coat', 'loading', 'mbike-trick', 
                    'pigs', 'shooting']
        val_data_dir = [val_flow_dir, img_dir, gt_dir]
        res = "480p"

    elif args.dataset == 'FBMS':
        basepath = args.basepath
        img_dir = args.basepath + '/FBMS/'
        gt_dir = args.basepath + '/Annotations/'    

        val_flow_dir = args.basepath + '/FBMS_val/Flows_gap1/'
        val_seq = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06', 
                    'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04', 
                    'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9', 
                    'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']
        val_img_dir = args.basepath + '/FBMS/'
        val_gt_dir =args.basepath + '/FBMS_annotation/'
        val_data_dir = [val_flow_dir, val_img_dir, val_gt_dir]
        with_gt = False
        pairs = [3, 6, -3, -6]

    elif args.dataset == 'STv2':
        basepath = args.basepath
        img_dir = basepath + '/STv2_img/JPEGImages/'
        gt_dir = basepath + '/STv2_gt&pred/STv2_gt/GroundTruth/'

        val_flow_dir = basepath + '/Flows_gap1/'
        val_seq = ['drift', 'birdfall', 'girl', 'cheetah', 'worm', 'parachute', 'monkeydog',
                    'hummingbird', 'soldier', 'bmx', 'frog', 'penguin', 'monkey', 'bird_of_paradise']
        val_data_dir = [val_flow_dir, img_dir, gt_dir]

    elif args.dataset == 'YTVOS':
        basepath = args.basepath
        img_dir = basepath + '/JPEGImages'
        gt_dir = basepath + '/Annotations'
        val_data_dir = None

    else:
        raise ValueError('Unknown Setting.')
    
    flow_dir = basepath
    data_dir = [flow_dir, img_dir, gt_dir]
    trn_dataset = Dataloader(data_dir=data_dir, dataset=args.dataset, resolution=resolution, gap=args.gap, to_rgb=args.flow_to_rgb, seq_length=args.num_frames,
                             train=True)
    if args.dataset == 'YTVOS':
        val_dataset = None
    else:
        val_dataset = Dataloader(data_dir=val_data_dir, dataset=args.dataset, resolution=resolution, gap=args.gap, to_rgb=args.flow_to_rgb,
                                train=False, val_seq=val_seq)
    in_out_channels = 3 if args.flow_to_rgb else 2
    use_flow = False if args.flow_to_rgb else True
    loss_scale = args.loss_scale
    ent_scale = args.ent_scale
    cons_scale = args.cons_scale

    
    return [trn_dataset, val_dataset, resolution, in_out_channels, use_flow, loss_scale, ent_scale, cons_scale]
