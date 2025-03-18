# revgcn
python revgnn_naive.py --model revgnn --num_layers 112 --hidden_channels 224 --dropout 0.2 --lr 0.001
torchrun --nnodes 1 --nproc_per_node 2 revgnn_pp_trainonall.py --dataset ogbn-arxiv --num_layers 112

# revgat
python revgnn_naive.py --model revgat --num_layers 5 --hidden_channels 256 --dropout 0.75 --lr 0.002 2>&1 | tee run1.log
torchrun --nnodes 1 --nproc_per_node 2 revgat_pp_trainonall.py --dataset ogbn-arxiv --num_layers 5 

TOKENIZERS_PARALLELISM=True torchrun --nnodes=1 --nproc_per_node=4 -m lm_workloads.lm_core.train_lm dataset ogbn-arxiv lm.train.use_gpt True

torchrun --nnodes 1 --nproc_per_node 4 --master_port 2923 revgnn_pp_trainonall_sync_data.py --dataset ogbn-arxiv --num_layers 112 --pid 1831335 1831336



[(Graph(num_nodes=84670, num_edges=301272,
      ndata_schemes={}
      edata_schemes={}), tensor([[-0.0579, -0.0525, -0.0726,  ...,  0.1734, -0.1728, -0.1401],
        [-0.1245, -0.0707, -0.3252,  ...,  0.0685, -0.3721, -0.3010],
        [-0.0802, -0.0233, -0.1838,  ...,  0.1099,  0.1176, -0.1399],
        ...,
        [-0.0246,  0.1469, -0.1367,  ...,  0.0759,  0.0569,  0.0302],
        [-0.0184, -0.1313, -0.2199,  ...,  0.1716, -0.1471, -0.1075],
        [-0.1053, -0.0035, -0.2914,  ...,  0.1819, -0.1032, -0.1746]],
       device='cuda:0')), (Graph(num_nodes=84670, num_edges=281661,
      ndata_schemes={}
      edata_schemes={}), tensor([[-0.1725, -0.1743, -0.1792,  ...,  0.1309,  0.0419,  0.0528],
        [-0.0446,  0.0463, -0.1204,  ...,  0.1383, -0.0432, -0.1850],
        [ 0.0727,  0.0513, -0.1931,  ...,  0.0736, -0.0631, -0.2323],
        ...,
        [-0.1917, -0.1209, -0.1911,  ...,  0.1034,  0.0231, -0.1891],
        [-0.3214, -0.0393, -0.0111,  ...,  0.0698, -0.0033, -0.2420],
        [-0.1512, -0.1247, -0.2214,  ...,  0.1203, -0.0628, -0.3163]],
       device='cuda:0'))]