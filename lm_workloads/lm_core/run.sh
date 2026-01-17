for dataset in 'ogbn-arxiv'
do 
TOKENIZERS_PARALLELISM=True torchrun --nnodes=1 --nproc_per_node=4 -m lm_workloads.lm_core.train_lm dataset ${dataset} 
done

# CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=True torchrun --nnodes=1 --nproc_per_node=1 -m lm_workloads.lm_core.train_lm dataset ogbn-arxiv >> lm_workloads/logs/${dataset}_lm.out
