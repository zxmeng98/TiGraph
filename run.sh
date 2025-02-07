# revgcn
python revgnn_naive.py --model revgnn --num_layers 112 --hidden_channels 224 --dropout 0.2 --lr 0.001
torchrun --nnodes 1 --nproc_per_node 2 revgnn_pp_trainonall.py --dataset ogbn-arxiv --num_layers 112

# revgat
python revgnn_naive.py --model revgat --num_layers 5 --hidden_channels 256 --dropout 0.75 --lr 0.002
torchrun --nnodes 1 --nproc_per_node 2 revgat_pp_trainonall.py --dataset ogbn-arxiv --num_layers 5
