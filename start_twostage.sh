python -W ignore -m torch.distributed.launch --nproc_per_node=4 --use_env \
		src/train.py \
        --basepath /path/to/training/data \
        --batch_size 32 \
        --seed 0 \
        --num_iterations 3 \
        --num_slots 16 \
        --num_instance 4 \
        --grad_iter 3000 \
        --num_train_steps 10000 \
        --lr 2e-5 \
        --gap 4 \
        --num_frames 4 \
        --entro_cons \
        --bi_cons \
        --output_path test_log \
        --dino_path dino_small_16.pth \
		--dataset YTVOS
