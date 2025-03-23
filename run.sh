# revgcn
python revgnn_naive.py --model revgnn --num_layers 112 --hidden_channels 224 --dropout 0.2 --lr 0.001
torchrun --nnodes 1 --nproc_per_node 2 revgnn_pp_trainonall.py --dataset ogbn-arxiv --num_layers 112
torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgnn_pp_trainonall_sync_lm_reorder.py --dataset ogbn-arxiv --num_layers 112 2>&1 | tee logs/revgcn_arxiv_sync_lm_acc.log

# revgat
python revgnn_naive.py --model revgat --num_layers 20 --hidden_channels 256 --dropout 0.75 --lr 0.002 2>&1 | tee logs/20revgat_arxiv_acc.log

torchrun --nnodes 1 --nproc_per_node 4 revgat_pp_trainonall.py --dataset ogbn-arxiv --num_layers 20 --hidden_channels 256 2>&1 | tee logs/revgat_arxiv_sync_lm_acc.log

( echo "Running command: torchrun --nnodes 1 --nproc_per_node 4 revgat_pp_trainonall.py --dataset ogbn-arxiv --num_layers 40 --hidden_channels 256"; echo ""; torchrun --nnodes 1 --nproc_per_node 4 revgat_pp_trainonall_sync_lm_reorder.py --dataset ogbn-arxiv --num_layers 20 --hidden_channels 256 ) 2>&1 | tee logs/20revgat_arxiv_sync_lm_acc.log

# resgen
( echo "torchrun --nnodes 1 --nproc_per_node 4 resgnn_pp_trainonall.py --num_layers 56 --hidden_channels 64 --dropout 0.5 --lr 0.01"; echo""; torchrun --nnodes 1 --nproc_per_node 4 resgnn_pp_trainonall.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01)  | tee logs/56resgnn_arxiv_acc.log

torchrun --nnodes 1 --nproc_per_node 2 resgnn_pp_trainonall_sync_lm_reorder.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01