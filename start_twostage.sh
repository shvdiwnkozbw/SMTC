python -W ignore -m torch.distributed.launch --nproc_per_node=8 --use_env \
		src/train.py \
        --basepath /path/to/training/data \
        --batch_size 64 \
        --seed 0 \
        --num_iterations 3 \
        --num_slots 16 \
        --num_instance 4 \
        --grad_iter 0 \
        --lr 1e-4 \
        --gap 4 \
        --num_frames 4 \
        --entro_cons \
        --bi_cons \
        --output_path test_log \
        --dino_path dino_small_16.pth \
		--dataset YTVOS
