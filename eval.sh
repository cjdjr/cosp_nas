#!/bin/bash
cd src/Seq2Seq
python -m torch.distributed.launch --nproc_per_node=4 search.py  --eval_only --no_output --no_tensorboard --load_path /data1/wangmr/SinglePathOneShot/outputs/exponential_sub_imgnet_20201115T062227/epoch-48.pt
# python -m torch.distributed.launch --nproc_per_node=2 search1.py --run_name 'SPOS_nobaseline' --no_tensorboard --eval_only --max-train-iters 1 --max-test-iters 1 
