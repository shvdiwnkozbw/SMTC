python -W ignore src/eval_multi.py \
        --basepath /path/to/evaluation/data \
        --batch_size 1 \
        --num_iterations 5 \
        --num_slots 16 \
        --num_instances 4 \
        --gap 4 \
        --hidden_dim 16 \
        --num_frames 4 \
        --output_path test_log \
        --dino_path dino_small_16.pth \
		--dataset DAVIS2017 \
        --resume_path /path/to/checkpoint
    
cd davis2017-evaluation

rm results/unsupervised/rvos/*.csv

python evaluation_method.py --task unsupervised --results_path results/unsupervised/rvos --set val --davis_path /mnt/data/mmlab_ie/qianrui/davis/DAVIS
