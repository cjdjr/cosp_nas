# COSP_NAS
Combinatorial Optimization for Single Path Neural Network Search

## TODO
- n_layers=4时候的显存占用量为什么更少？
- 为什么训练的时候会朝着反方向去训练？
    - 猜测可能是每个batch采样的problem由于数量比较少，导致可能本身带来的结果就差
    - check 每个epoch的problem dataset
    - check evaluator是否一致
    - check baseline

- 将训练过程的input设置为不定长，看效果
- 重写decoder部分，参考transformer decoder，主要是decoder阶段要拿到之前所有decode的信息

