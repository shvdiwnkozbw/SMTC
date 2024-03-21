rm -r save_davis_semi
rm -r convert_davis_semi

python src/eval_prop.py --filelist src/eval_utils/davis_vallist.txt --resume /path/to/checkpoint --save-path save_davis_semi/ --topk 7 --videoLen 20 --radius 7 --temperature 0.2 --cropSize -1 --gpu-id 0

# python src/eval_prop.py --filelist src/eval_utils/davis_vallist.txt --resume /path/to/checkpoint --save-path save_davis_semi/ --topk 28 --videoLen 20 --radius 35 --temperature 0.2 --cropSize -1 --gpu-id 0

python src/eval_utils/convert_davis.py --in_folder save_davis_semi/ --out_folder convert_davis_semi/ --dataset /path/to/evaluation/data

python davis2017-evaluation/evaluation_method.py --task semi-supervised --results_path convert_davis_semi/ --set val --davis_path /path/to/evaluation/data
