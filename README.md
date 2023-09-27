# SMTC

Code of ICCV paper [Semantics Meets Temporal Correspondence: Self-supervised Object-centric Learning in Videos](https://arxiv.org/abs/2308.09951). 

We propose a two-stage slot attention design, named semantic-aware masked slot attention, to jointly utilize rich semantics and fine-grained temporal correspondence in videos to distill temporally coherent object-centric representations.

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

We use other video dataset for evaluation on unsupervised video object discovery as well as label propagation tasks, including DAVIS-2017, SegTrack-v2, FMBS-59, JHMDB, VIP. These dataset are formulated in the similar manner, and here we take DAVIS-2017 as an example.
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

## Evaluation

We follow the conventional evaluation protocols in previous works on unsupervised video object discovery, e.g., [OCLR](https://github.com/jyxarthur/oclr_model), and label propagation, e.g., [CRW](https://ajabri.github.io/videowalk/).
