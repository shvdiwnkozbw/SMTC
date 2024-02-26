rm -r save_davis_semi
rm -r convert_davis_semi

python src/eval_prop.py --filelist src/eval_utils/davis_vallist.txt --resume /mnt/data/mmlab_ie/qianrui/MG_factory/test_log/2024_02_24_19_34-DAVIS-rgb-dim_16gap_4_t_1_o_0-lr_2e-05-bs_32/model/checkpoint_7000_iou_0.pth --save-path save_davis_semi/ --topk 40 --videoLen 20 --radius 40 --temperature 0.05 --cropSize -1 --gpu-id 0

python src/eval_utils/convert_davis.py --in_folder save_davis_semi/ --out_folder convert_davis_semi/ --dataset /mnt/data/mmlab_ie/qianrui/davis/DAVIS

python davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path convert_davis_semi/ --set val --davis_path /mnt/data/mmlab_ie/qianrui/davis/DAVIS
