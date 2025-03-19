# revgcn
python revgnn_naive.py --model revgnn --num_layers 112 --hidden_channels 224 --dropout 0.2 --lr 0.001
torchrun --nnodes 1 --nproc_per_node 2 revgnn_pp_trainonall.py --dataset ogbn-arxiv --num_layers 112

# revgat
python revgnn_naive.py --model revgat --num_layers 5 --hidden_channels 256 --dropout 0.75 --lr 0.002 2>&1 | tee run1.log
torchrun --nnodes 1 --nproc_per_node 2 revgat_pp_trainonall.py --dataset ogbn-arxiv --num_layers 5 

TOKENIZERS_PARALLELISM=True torchrun --nnodes=1 --nproc_per_node=4 -m lm_workloads.lm_core.train_lm dataset ogbn-arxiv lm.train.use_gpt True



torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgnn_pp_trainonall_sync_data_reorder.py --dataset ogbn-arxiv --num_layers 112 --pid 61857 61858 61859 61860

