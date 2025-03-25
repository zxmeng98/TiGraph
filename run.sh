# ------------------------------------ LM ------------------------------------

TOKENIZERS_PARALLELISM=True torchrun --nnodes=1 --nproc_per_node=4 -m lm_workloads.lm_core.train_lm dataset ogbn-arxiv lm.train.batch_size 9 lm.train.grad_acc_steps 1



# ------------------------------------ GNN ------------------------------------
# revgcn
python revgnn_naive.py --model revgnn --num_layers 112 --hidden_channels 224 --dropout 0.2 --lr 0.001

torchrun --nnodes 1 --nproc_per_node 4 revgnn_pp_trainonall.py --dataset pubmed --num_layers 448 --hidden_channels 80 --epochs 1000 --dropout 0.1 --lr 0.001 --bs 19717 --mb_size 19717

( echo "Running command: torchrun --nnodes 1 --nproc_per_node 4 revgnn_pp_trainonall_sync_lm_reorder.py --dataset ogbn-arxiv --num_layers 112"; echo ""; torchrun --nnodes 1 --nproc_per_node 4 revgnn_pp_trainonall_sync_lm_reorder.py --dataset ogbn-arxiv --num_layers 112 ) 2>&1 | tee logs/revgcn_arxiv_sync_lm_acc.log
torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgnn_pp_trainonall_sync_lm_reorder.py --dataset ogbn-arxiv --num_layers 112 2>&1 | tee logs/revgcn_arxiv_sync_lm_acc.log

# revgat
python revgnn_naive.py --model revgat --num_layers 20 --hidden_channels 256 --dropout 0.75 --lr 0.002 2>&1 | tee logs/20revgat_arxiv_acc.log

torchrun --nnodes 1 --nproc_per_node 4 revgat_pp_trainonall.py --dataset ogbn-arxiv --num_layers 20 --hidden_channels 256 2>&1 | tee logs/revgat_arxiv_sync_lm_acc.log

( echo "Running command: torchrun --nnodes 1 --nproc_per_node 4 revgat_pp_trainonall.py --dataset ogbn-arxiv --num_layers 40 --hidden_channels 256"; echo ""; torchrun --nnodes 1 --nproc_per_node 4 revgat_pp_trainonall_sync_lm_reorder.py --dataset ogbn-arxiv --num_layers 20 --hidden_channels 256 ) 2>&1 | tee logs/20revgat_arxiv_sync_lm_acc.log

torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgat_pp_trainonall_sync_lm_reorder.py --dataset ogbn-arxiv --num_layers 20 --hidden_channels 256 --pid 1653804 1653805 1653806 1653807

# resgen
orchrun --nnodes 1 --nproc_per_node 4 resgnn_pp_trainonall.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01

( echo "torchrun --nnodes 1 --nproc_per_node 4 resgnn_pp_trainonall_sync_lm_reorder.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01"; echo""; torchrun --nnodes 1 --nproc_per_node 4 resgnn_pp_trainonall_sync_lm_reorder.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01)  | tee logs/56resgnn_arxiv_sync_lm_acc.log

torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 resgnn_pp_trainonall_sync_lm_reorder.py --num_layers 56 --hidden_channels 128 --dropout 0.5 --lr 0.01 --pid 1678330 1678331 1678332 1678333