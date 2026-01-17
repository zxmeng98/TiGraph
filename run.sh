# ------------------------------------ LM ------------------------------------

TOKENIZERS_PARALLELISM=True torchrun --nnodes=1 --nproc_per_node=4 -m lm_workloads.lm_core.train_lm dataset pubmed

# ------------------------------------ GNN ------------------------------------
# revgcn
python revgnn_naive.py --model revgnn --num_layers 112 --hidden_channels 224 --dropout 0.2 --lr 0.001

dataset=ogbn-arxiv
layers=112
hidden=224
torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgnn_pp_trainonall.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --epochs 500 --dropout 0.2 --lr 0.001 --bs 169340 --mb_size 42335 --pid 2875996 2875997 2875998 2875999

torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgnn_pp_trainonall_sync_lm_reorder.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --epochs 1000 --dropout 0.2 --lr 0.001 --bs 169340 --mb_size 42335 --pid 2855785 2855786 2855787 2855788


# revgat
python revgnn_naive.py --model revgat --num_layers 20 --hidden_channels 256 --dropout 0.75 --lr 0.002 2>&1 | tee logs/20revgat_arxiv_acc.log

dataset=ogbn-arxiv
layers=20
hidden=256
for mb_size in 84670
do
torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgat_pp_trainonall.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --bs 169340 --mb_size $mb_size --dropout 0.5 --lr 0.002 --epochs 2000 
done

torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgat_pp_trainonall_sync_lm_reorder.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --bs 169328 --mb_size 84670 --dropout 0.5 --lr 0.002 --epochs 500 --pid 4058272 4058273 4058274 4058275


( echo "Running command: torchrun --nnodes 1 --nproc_per_node 4 revgat_pp_trainonall.py --dataset ogbn-arxiv --num_layers 40 --hidden_channels 256"; echo ""; torchrun --nnodes 1 --nproc_per_node 4 revgat_pp_trainonall_sync_lm_reorder.py --dataset ogbn-arxiv --num_layers 20 --hidden_channels 256 ) 2>&1 | tee logs/20revgat_arxiv_sync_lm_acc.log


# resgen
torchrun --nnodes 1 --nproc_per_node 4 resgnn_pp_trainonall.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01

( echo "torchrun --nnodes 1 --nproc_per_node 4 resgnn_pp_trainonall_sync_lm_reorder.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01"; echo""; torchrun --nnodes 1 --nproc_per_node 4 resgnn_pp_trainonall_sync_lm_reorder.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01)  | tee logs/56resgnn_arxiv_sync_lm_acc.log

dataset=ogbn-arxiv
layers=56
hidden=128
torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 resgnn_pp_trainonall_sync_lm_reorder.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --dropout 0.5 --lr 0.01 --epochs 1000 --bs 54025 --mb_size 54025 >> logs/${layers}-${hidden}resgnn_${dataset}_sync_lm_acc.log

torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 resgnn_pp_trainonall_sync_lm_reorder.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --dropout 0.5 --lr 0.01 --epochs 1000 --bs 54025 --mb_size 54025 --pid 4065547 4065548 4065549 4065550