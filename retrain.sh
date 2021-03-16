
#!/bin/bash
# cd src/Evaluation/data/'(2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)'
# python3 -m torch.distributed.launch --nproc_per_node=4 train.py
root=$PWD
cd src/Evaluation/data/'(2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)'
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --train-dir $root/data/train --val-dir $root/data/val --learning-rate 0.125 --save ./models_lr_0.125
