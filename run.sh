torchrun --nnodes 1 --nproc_per_node 4 example_bert.py
torchrun --nnodes 1 --nproc_per_node 2 example_gnn.py
torchrun --nnodes 1 --nproc_per_node 2 torchpipe.py