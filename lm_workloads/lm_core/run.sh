for dataset in 'cora' 'pubmed' 'ogbn-arxiv' 'ariv_2023' 'ogbn-products'
do
    for seed in 0 1 2 3
    do
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset $dataset seed $seed >> ${dataset}_lm.out
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset $dataset seed $seed lm.train.use_gpt True  >> ${dataset}_lm2.out
    done
    python -m core.trainEnsemble dataset $dataset gnn.model.name MLP >> ${dataset}_mlp.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name GCN >> ${dataset}_gcn.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name SAGE >> ${dataset}_sage.out
    python -m core.trainEnsemble dataset $dataset gnn.model.name RevGAT gnn.train.lr 0.002 gnn.train.dropout 0.5 >> ${dataset}_revgat.out
done


WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python -m lm_workloads.lm_core.train_lm dataset ogbn-arxiv 2>&1 | tee ./lm_workloads/deberta-base.log

WANDB_DISABLED=True TOKENIZERS_PARALLELISM=True CUDA_VISIBLE_DEVICES=0 python -m lm_workloads.lm_core.train_lm dataset ogbn-arxiv lm.train.use_gpt True

WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lm_workloads.lm_core.train_lm dataset ogbn-arxiv lm.train.use_gpt True

NCCL_MAX_NCHANNELS=1 TOKENIZERS_PARALLELISM=True torchrun --nnodes=1 --nproc_per_node=4 -m lm_workloads.lm_core.train_lm dataset ogbn-arxiv lm.train.use_gpt True
