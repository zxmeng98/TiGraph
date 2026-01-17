
torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgnn_pp_trainonall_sync_lm_reorder.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --epochs 1000 --dropout 0.2 --lr 0.001 --bs 54024 --mb_size 13506 --pid 1397973

torchrun --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr "192.168.1.40" \
    --master_port 2923 \
    revgnn_pp_trainonall_sync_lm_reorder.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --dropout 0.5 --lr 0.01 --bs 54025 --mb_size 54025

torchrun --nproc_per_node 4 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr "192.168.1.40" \
    --master_port 2923 \
    revgnn_pp_trainonall_sync_lm_reorder.py --dataset $dataset --num_layers $layers --hidden_channels $hidden --dropout 0.5 --lr 0.01 --bs 54025 --mb_size 54025