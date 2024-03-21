from operator import index

import os
import time
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import vos

import eval_utils as utils
import eval_utils.test_utils as test_utils
import cv2
import einops
import sys
from model.model_twostage import SlotAttentionAutoEncoder

def main(args, vis):
    model = SlotAttentionAutoEncoder(resolution=(256, 256),
                                num_slots=16,
                                num_instances=4,
                                in_channels=3, 
                                out_channels=3,
                                hid_dim=16,
                                iters=3,
                                dino_path='dino_small_16.pth')

    args.mapScale = 320 // np.array([20, 20])
    # args.mapScale = 320 // np.array([40, 40])

    dataset = vos.VOSDataset(args)
    val_loader = torch.utils.data.DataLoader(dataset,
        batch_size=int(args.batchSize), shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load checkpoint.
    if os.path.isfile(args.resume):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        del checkpoint
    
    model.eval()
    model = model.to(args.device)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    with torch.no_grad():
        test_loss = test(val_loader, model, args)


def test(loader, model, args):
    n_context = args.videoLen
    D = None    # Radius mask
    
    for vid_idx, (imgs, imgs_orig, lbls, lbls_orig, lbl_map, meta) in enumerate(loader):
        t_vid = time.time()
        imgs = imgs.to(args.device)
        B, N = imgs.shape[:2]
        assert(B == 1)

        H, W = imgs.shape[-2:]
        new_h = H // args.mapScale[0] * args.mapScale[0]
        new_w = W // args.mapScale[1] * args.mapScale[1]
        imgs = torch.nn.functional.interpolate(imgs, size=(3, new_h, new_w), mode='trilinear')

        print('******* Vid %s (%s frames) *******' % (vid_idx, N))
        with torch.no_grad():
            t00 = time.time()

            ##################################################################
            # Compute image features (batched for memory efficiency)
            ##################################################################
            bsize = 16   # minibatch size for computing features
            feats = []
            for b in range(0, imgs.shape[1], bsize):
                feat, cls_token, _, key = model.encoder(imgs[:, b:b+bsize][0].to(args.device))
                feat = feat.permute(1, 0, 2, 3).contiguous().unsqueeze(0) # b c t h w
                cls_token = cls_token.permute(1, 0).unsqueeze(0) # b c t
                B, C, T, H, W = feat.shape
                feat = torch.nn.functional.interpolate(feat, (T, lbls.shape[2], lbls.shape[3]), mode='trilinear')
                feats.append(feat.cpu())
            feats = torch.cat(feats, dim=2)

            if not args.no_l2:
                feats = torch.nn.functional.normalize(feats, dim=1)

            print('computed features', time.time()-t00)

            ##################################################################
            # Compute affinities
            ##################################################################
            torch.cuda.empty_cache()
            t03 = time.time()
            
            # Prepare source (keys) and target (query) frame features
            key_indices = test_utils.context_index_bank(n_context, args.long_mem, N - n_context)
            key_indices = torch.cat(key_indices, dim=-1)
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]
            # Make spatial radius mask TODO use torch.sparse
            restrict = utils.MaskedAttention(args.radius, flat=False)
            D = restrict.mask(*feats.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D==0] = -1e10; D[D==1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            print('computing affinity')
            Ws, Is = test_utils.mem_efficient_batched_affinity(query, keys, model, D, 
                        args.temperature, args.topk, args.long_mem, args.device)
            # print(len(Ws), Ws[0].shape, len(Is), Is[0].shape, query.shape, keys.shape, key_indices.shape)
            # Ws, Is = test_utils.batched_affinity(query, keys, D, 
            #             args.temperature, args.topk, args.long_mem, args.device)

            if torch.cuda.is_available():
                print(time.time()-t03, 'affinity forward, max mem', torch.cuda.max_memory_allocated() / (1024**2))

            ##################################################################
            # Propagate Labels and Save Predictions
            ###################################################################

            maps, keypts = [], []
            lbls[0, n_context:] *= 0 
            lbl_map, lbls = lbl_map[0], lbls[0]

            for t in range(key_indices.shape[0]):
                # Soft labels of source nodes
                ctx_lbls = lbls[key_indices[t]].to(args.device)
                ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

                # Weighted sum of top-k neighbours (Is is index, Ws is weight) 
                pred = (ctx_lbls[:, Is[t]] * Ws[t].to(args.device)[None]).sum(1)
                pred = pred.view(-1, *feats.shape[-2:])
                pred = pred.permute(1,2,0)
                
                if t > 0:
                    lbls[t + n_context] = pred
                else:
                    pred = lbls[0]
                    lbls[t + n_context] = pred

                if args.norm_mask:
                    pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                    pred[:, :, :] /= pred.max(-1)[0][:, :, None]

                # Save Predictions            
                cur_img = imgs_orig[0, t + n_context].permute(1,2,0).numpy() * 255
                _maps = []

                outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))

                heatmap, lblmap, heatmap_prob = test_utils.dump_predictions(
                    pred.cpu().numpy(),
                    lbl_map, cur_img, outpath)

                _maps += [heatmap, lblmap, heatmap_prob]
                maps.append(_maps)

                if args.visdom:
                    [vis.image(np.uint8(_m).transpose(2, 0, 1)) for _m in _maps]

            if len(keypts) > 0:
                coordpath = os.path.join(args.save_path, str(vid_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)
            
            if vis:
                wandb.log({'blend vid%s' % vid_idx: wandb.Video(
                    np.array([m[0] for m in maps]).transpose(0, -1, 1, 2), fps=12, format="gif")})  
                wandb.log({'plain vid%s' % vid_idx: wandb.Video(
                    imgs_orig[0, n_context:].numpy(), fps=4, format="gif")})  
                
            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (vid_idx, time.time() - t_vid))


if __name__ == '__main__':
    args = utils.arguments.test_args()

    args.imgSize = args.cropSize
    print('Context Length:', args.videoLen, 'Image Size:', args.imgSize)
    print('Arguments', args)

    vis = None
    if args.visdom:
        import visdom
        import wandb
        vis = visdom.Visdom(server=args.visdom_server, port=8095, env='main_davis_viz1'); vis.close()
        wandb.init(project='palindromes', group='test_online')
        vis.close()

    main(args, vis)
