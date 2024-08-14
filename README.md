# Semantics Meets Temporal Correspondence: Self-supervised Object-centric Learning in Videos

Official Code of ICCV 2023 paper [Semantics Meets Temporal Correspondence: Self-supervised Object-centric Learning in Videos](https://arxiv.org/abs/2308.09951). 

We propose a two-stage slot attention design, named semantic-aware masked slot attention, to jointly utilize rich semantics and fine-grained temporal correspondence in videos to distill temporally coherent object-centric representations.

## Updates

There is an evaluation bug for ViT backbone in label propagation. Specifically, some images in DAVIS-2017 have height/width dimensions that are not multiples of the patch size, so that there will be information loss in the patch embedding process. To this end, we apply resize operation to guarantee that the height/width of all images are multiples of the patch size. And through this processing, the performance of ViT backbone is much higher than reported in the original DINO paper. And the gap between the DINO model and our tuned model becomes negligible both on the original size and interpolated size. And we believe there is much potential for ViT based model to achieve superior correspondence results than CNN based ones. We have also updated a new version on [arXiv](https://arxiv.org/abs/2308.09951).

## Requirements

- Python 3.9
- PyTorch 1.9
- torchvision

## Prepare Dataset

#### YouTube-VOS

Download YouTube-VOS dataset, then unzip the video frames, annotations as follows:
```
YTVOS
|----train
|--------JPEGImages
|------------video 1
|----------------00000.jpg
|----------------00001.jpg
    		...
|----------------0000n.jpg
    	...
|------------video n
    ...
|--------Annotations
|------------video 1
|----------------00000.png
|----------------00001.png
    		...
|----------------0000n.png
    	...
|------------video n
```

Since we only use YouTube-VOS data for fully self-supervised training, we do not access the annotations in this repo.

#### Evaluation Data

We use other video datasets for evaluation on unsupervised video object discovery as well as label propagation tasks, including DAVIS-2017, SegTrack-v2, FMBS-59, JHMDB, VIP. These datasets are formulated in the similar manner, and here we take DAVIS-2017 as an example.
```
DAVIS
|----JPEGImages
|--------480p
|------------video 1
|----------------00000.jpg
|----------------00001.jpg
    		...
|----------------0000n.jpg
    	...
|------------video n
    ...
|----Annotations
|--------480p
|------------video 1
|----------------00000.png
|----------------00001.png
    		...
|----------------0000n.png
    	...
|------------video n
```
We report the evaluation results on the validation set.

## Training

By changing the basepath in `start_twostage.sh` to adjust the training data path. Note that the batchsize and workers hyper-parameters are for each GPU process in distributed training.

```
python -W ignore -m torch.distributed.launch --nproc_per_node=8 --use_env \
		src/train.py \
	--basepath /path/to/training/data \
	--batch_size 64 \
	--seed 0 \
	--num_iterations 3 \
	--num_slots 16 \
	--num_instance 4 \
	--lr 1e-4 \
	--gap 4 \
	--num_frames 4 \
	--entro_cons \
	--bi_cons \
	--output_path test_log \
	--dino_path dino_small_16.pth \
	--dataset YTVOS
```
It is flexible to adjust the number of slots (semantic centers) `num_slots`, the number of sampled slots (instances per semantic) `num_instances`, input frame length `num_frame` and rate `gap` to explore the impact of these import hyper-parameters.

The original version sets a strict standard to filter the valid instances to ensure the high quality instance samples in training. Hence, it requires very long training iterations to let the model satisfy this standard. To address this limitation, we update a new version by relaxing the standard for valid instance sample filtering, so that the training requires much fewer computational resources and reaches a tradeoff between performance and efficiency. The newly introduced strategies include using lower threshold to filter valid activation area, freezing DINO pretrained ViT weights in the early stages of training, etc. This efficient version only requires around 7k iterations to achieve promising results.

```
python -W ignore -m torch.distributed.launch --nproc_per_node=4 --use_env \
		src/train.py \
        --basepath /path/to/training/data \
        --batch_size 32 \
        --seed 0 \
        --num_iterations 3 \
        --num_slots 16 \
        --num_instance 4 \
        --grad_iter 3000 \
        --lr 2e-5 \
        --gap 4 \
        --num_frames 4 \
        --entro_cons \
        --bi_cons \
        --output_path test_log \
        --dino_path dino_small_16.pth \
        --dataset YTVOS
```

## Evaluation

We follow the conventional evaluation protocols in previous works on unsupervised video object discovery, e.g., [OCLR](https://github.com/jyxarthur/oclr_model), and label propagation, e.g., [CRW](https://ajabri.github.io/videowalk/).

Here, we provide two examples respectively on semi-supervised label propagation task on DAVIS-2017 and unsupervised multiple object discovery on DAVIS-2017-Unsupervised.

For multiple object discovery task, run `eval_multi.sh`, it produces the candidate object masks and calculates the J&F score. For label propagation task, run `eval_prop.sh`, it generates the semi-supervised segmentation results and outputs the J&F score. Note that in this script we provide two settings. The first is to perform label propagation on the original ViT-S/16 feature with downsample ratio of 16. The second is to first interpolate the ViT features into feature maps with downsample ratio of 8, the same as the prevalent CNN based models. The provided reference hyper-parameters may not be optimal, you are free to tune them for better performance.

## Pretrained Model

We also provide a trained ViT-S/16 model with `num_slots=16` and `num_instances=4`. The model weight is available at this google drive [link](https://drive.google.com/file/d/162dtjPXQ2r4lghg6W5Vu8x2lRj0EJmtU/view?usp=drive_link). This model is trained with the newly updated code with `start_twostage.sh`. This model is trained with a relaxed valid instance filtering standard, requiring fewer computations, achieving 64.0/67.6 J&F score on DAVIS-2017 Semi-supervised, 44.8 J&F score on DAVIS-2017 unsupervised multiple object discovery.

## ✒Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@inproceedings{qian2023semantics,
  title={Semantics meets temporal correspondence: Self-supervised object-centric learning in videos},
  author={Qian, Rui and Ding, Shuangrui and Liu, Xian and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16675--16687},
  year={2023}
}
```
