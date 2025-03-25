# Transductive train: training sees whole graph structure, given a few labeled nodes, to classify the rest of the nodes.
import argparse
import sys
import dgl
import dgl.nn as dglnn
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import random
import numpy as np
import os
from GNNs.RevGNN.revgcn_pp import RevGCN
from utils.dataset import get_dataset
import torch.distributed as dist
from ogb.nodeproppred import DglNodePropPredDataset
from pipelining import pipeline, SplitPoint, PipelineStage
from pipelining.schedules_interleave_dp_reorder import ScheduleGPipe
from pipelining._utils import partition_uniform
from pipelining.initialize import (
    init_distributed, 
    init_small_workload_pid, 
    get_pipeline_parallel_rank,
    get_device,
    get_pipeline_parallel_group,
    get_pipeline_parallel_world_size,
    )
from utils.dataset import intersection, DataProcess, reformat_graph, split_mb_in_advance
from od_execution.client import send_signal


global rank, device, pp_group, stage_index, num_stages
def init_pp():
    global rank, device, pp_group, stage_index, num_stages
    init_distributed()
    rank = get_pipeline_parallel_rank()
    device = get_device()
    pp_group = get_pipeline_parallel_group()
    stage_index = rank
    num_stages = get_pipeline_parallel_world_size()


def manual_model_split(args, model, example_input_microbatch) -> PipelineStage:
    parts = partition_uniform(args.num_layers, num_stages)
    start = parts[stage_index]
    stop = parts[stage_index + 1]
    print(f'stage={stage_index} layers={start} - {stop}')
    if stage_index == 0:
        # prepare the first stage model
        model.gcns = model.gcns[start:stop]
        model.last_norm = None
        model.dropout_layer = None
        model.node_pred_linear = None
        stage_input_microbatch = example_input_microbatch

    elif stage_index == num_stages - 1:
        # prepare the second stage model
        model.node_features_encoder = None
        model.gcns = model.gcns[start:stop]
        feature_input_microbatch = torch.randn(example_input_microbatch[1].shape[0], args.hidden_channels)
        stage_input_microbatch = (example_input_microbatch[0], feature_input_microbatch)

    else:
        model.node_features_encoder = None
        model.gcns = model.gcns[start:stop]
        model.last_norm = None
        model.dropout_layer = None
        model.node_pred_linear = None
        feature_input_microbatch = torch.randn(example_input_microbatch[1].shape[0], args.hidden_channels)
        stage_input_microbatch = (example_input_microbatch[0], feature_input_microbatch)
        
    stage = PipelineStage(
      model,
      stage_index,
      num_stages,
      device,
      input_args=stage_input_microbatch,
   )
    return stage


def evaluate(args, preprocessed_batches, batch_idxes, split_idx, stage, schedule):
    valid_output, test_output = [], []
    valid_labels, test_labels = [], []
    stage.submod.eval()
    for i in range(len(batch_idxes)):
        batch_data = preprocessed_batches[i]
        args_split = batch_data['args_split']
        kwargs_split = batch_data['kwargs_split']
        chunked_sg_ori_node_idxes = batch_data['chunked_sg_ori_node_idxes']
        labels_split = batch_data['reordered_targets']
        # batch_idx = batch_idxes[i].tolist()
        batch_idx = torch.cat(chunked_sg_ori_node_idxes).tolist()
        labels_i = torch.cat(labels_split)

        if rank == 0:
            schedule.step(args_split, kwargs_split, chunked_sg_ori_node_idxes, split_idx=split_idx['valid'])
        elif rank == num_stages - 1:
            output = schedule.step(args_split, kwargs_split, chunked_sg_ori_node_idxes, target=labels_split, split_idx=split_idx['valid'])   
            mapper = {node: idx for idx, node in enumerate(batch_idx)}
            inter_valid_node = intersection(batch_idx, split_idx['valid'].tolist())
            inter_test_node = intersection(batch_idx, split_idx['test'].tolist())
            local_valid_idx = [mapper[node] for node in inter_valid_node]
            local_test_idx = [mapper[node] for node in inter_test_node]

            valid_output.append(output[local_valid_idx])
            test_output.append(output[local_test_idx])
            valid_labels.append(labels_i[local_valid_idx])
            test_labels.append(labels_i[local_test_idx])
        else:
            schedule.step(args_split, kwargs_split, chunked_sg_ori_node_idxes, split_idx=split_idx['valid'])
    
    if rank == num_stages - 1:
        # Need to make sure labels and outputs match. 
        # 1. Save local labels in each batch and use them compute. 
        # 2. Can reorder outputs to match the original labels[split_idx['valid']].
        valid_output = torch.cat(valid_output, 0)
        test_output = torch.cat(test_output, 0)
        valid_labels = torch.cat(valid_labels, 0)
        test_labels = torch.cat(test_labels, 0)

        _, valid_indices = torch.max(valid_output, dim=1)
        _, test_indices = torch.max(test_output, dim=1)

        assert len(valid_indices) == len(valid_labels)
        assert len(test_indices) == len(test_labels)

        valid_correct = torch.sum(valid_indices == valid_labels)
        test_correct = torch.sum(test_indices == test_labels)

        valid_acc = valid_correct.item() * 100.0 / len(valid_indices)
        test_acc = test_correct.item() * 100.0 / len(test_indices)
        return (valid_acc, test_acc)
        
    

if __name__ == "__main__":
    init_pp()
      
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        help='dataset name (default: ogbn-proteins)')
    
    # Training & eval settings
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--bs', type=int, default=169340,
                       help='Number of microbatches.')
    parser.add_argument('--mb_size', type=int, default=42335, # 42335, 84670
                       help='Number of microbatches.')

    # Pipeline pearallel 
    parser.add_argument('--rank', type=int, default=None,
                       help='rank passed from distributed launcher.')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    parser.add_argument('--world-size', type=int, default=None,
                       help='world size of sequence parallel group.')
    parser.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'ccl'],
                       help='Which backend to use for distributed training.')
    parser.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    parser.add_argument('--pipeline-parallel-size', type=int, default=4,
                       help='Enable pipeline parallel.')
    
    # Interleaved workload
    parser.add_argument('--pid', nargs='+', type=int, default=None, help='PID of the small workload.')
    parser.add_argument('--lm_model', type=str, default='deberta-base',
                        help='deberta-base, ')
    

    # Model
    parser.add_argument('--gnn_model', type=str, default='revgnn',
                        help='gcn backbone [revgcn, resgnn, revgat]')
    parser.add_argument('--backbone', type=str, default='rev',
                        help='gcn backbone [deepergcn, weighttied, deq, rev]')
    parser.add_argument('--group', type=int, default=2,
                        help='num of groups for rev gnns')
    parser.add_argument('--num_layers', type=int, default=112,
                        help='the number of layers of the networks')
    parser.add_argument('--mlp_layers', type=int, default=2,
                        help='the number of layers of mlp in conv')
    parser.add_argument('--in_size', type=int, default=50,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--out_size', type=int, default=3,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--hidden_channels', type=int, default=224,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--gcn_aggr', type=str, default='max',
                        help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
    parser.add_argument('--norm', type=str, default='layer',
                        help='the type of normalization layer')
    
    # Learnable parameters
    parser.add_argument('--t', type=float, default=1.0,
                        help='the temperature of SoftMax')
    parser.add_argument('--p', type=float, default=1.0,
                        help='the power of PowerMean')
    parser.add_argument('--y', type=float, default=0.0,
                        help='the power of degrees')
    parser.add_argument('--learn_t', action='store_true')
    parser.add_argument('--learn_p', action='store_true')
    parser.add_argument('--learn_y', action='store_true')

    # message norm
    parser.add_argument('--msg_norm', action='store_true')
    parser.add_argument('--learn_msg_scale', action='store_true')
    # encode edge in conv
    parser.add_argument('--conv_encode_edge', action='store_true')
    # if use one-hot-encoding node feature
    parser.add_argument('--use_one_hot_encoding', action='store_true')
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize small workload PID
    pid = None
    if args.pid is not None:
        if len(args.pid) > rank:  # This rank has a specified PID
            pid = args.pid[rank]
        elif len(args.pid) == 1 and rank == 0:  # Only first rank gets the single PID
            pid = args.pid[0]

    init_small_workload_pid(pid)
    small_workload_pid = pid

    # Load and preprocess dataset
    g, split_idx, features, labels, num_classes = get_dataset(args.dataset)
    LM_emb_path = f"./lm_workloads/prt_lm/{args.dataset}/microsoft/deberta-base-seed0.emb"
    if os.path.exists(LM_emb_path):
        if rank == 0:
            print("Loading trained LM features (title and abstract) ...")
            print(f"LM_emb_path: {LM_emb_path}")
        features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                        dtype=np.float16,
                        shape=(g.num_nodes(), 768)))
        ).to(torch.float32)
        g.ndata['feat'] = features

    data_processed = DataProcess(args, g, split_idx, features, labels)
    args.in_size = data_processed.features.shape[1]
    args.out_size = num_classes
    
    if rank == 0:
        print(f"{args.dataset} load and processed successfully")
        print(f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}")
    
    if data_processed.num_batches > 1:
        batch_g = dgl.node_subgraph(g, data_processed.batch_nodes[0])
    else:
        batch_g = g
    
    num_microbatches = data_processed.batch_nodes[0].shape[0] // args.mb_size
    batch_local_idxes = torch.arange(batch_g.num_nodes())
    microbatch_idxes = torch.tensor_split(batch_local_idxes, num_microbatches)
    microbatch_g = dgl.node_subgraph(batch_g, microbatch_idxes[0])
    
    microbatch_g.ndata.clear()
    microbatch_g.edata.clear()
    example_input_microbatch = (microbatch_g, data_processed.features[microbatch_idxes[0]])

    if rank == 0:
        print(f"Num of train batches: {data_processed.num_batches}, num of train microbatches: {num_microbatches}, microbatch size: {args.mb_size}")
    
    # Pack batch graph trained in each iter beforehand
    packed_batch = data_processed.pack_batch(data_processed.features)

    # Create GCN model
    model = RevGCN(args)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNumber of parameters: {trainable_params}")

    stage = manual_model_split(args, model, example_input_microbatch)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(stage.submod.parameters(), lr=args.lr)
    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=loss_fcn)

    # Convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)      

    # After pp workload start up, pause the small workload till the bubble
    if rank == 0 and small_workload_pid is not None:
        send_signal(small_workload_pid, "pause")

    # Split in advance
    preprocessed_batches = split_mb_in_advance(data_processed.num_batches, schedule, packed_batch, device, k=4)


    # Model training
    if rank == 0:
        print("Training...")

    timestamp0 = time.time()
    local_time = time.localtime(timestamp0)
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    ms = int((timestamp0 - int(timestamp0)) * 1000)
    formatted_time = f"{time_str}.{ms:03d}"
    print(f"{args.gnn_model} start time: {formatted_time}")
        
    # Training loop
    loss_list, val_acc_list, test_acc_list, epoch_time_list = [], [], [], [] 

    pp_profile = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            skip_first=5, wait=1, warmup=1, active=3, repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./tensorboard_trace/GNN-PP-LM-DP/gnn-pp-reorder/"
        ),
        # f"./tensorboard_trace/revgnn_pp{num_stages}_stage{stage_index}_iter/"
        # with_stack=True,
        with_modules=True,
        # profile_memory=True,
    )
    # pp_profile.start()

    delta_sync = 20
    last_written_rows = 0
    for epoch in range(args.epochs):
        # if (epoch+1) == delta_sync:
        #     packed_batch_from_lm, last_written_rows = data_processed.check_load_from_lm("TA", last_written_rows)
        #     if packed_batch_from_lm is not None:
        #         packed_batch = packed_batch_from_lm
        optimizer.zero_grad()
        stage.submod.train()
        t0 = time.time()
        for i in range(data_processed.num_batches):
            # Get preprocessed data for this batch
            batch_data = preprocessed_batches[i]
            args_split = batch_data['args_split']
            kwargs_split = batch_data['kwargs_split']
            chunked_sg_ori_node_idxes = batch_data['chunked_sg_ori_node_idxes']
            labels_split = batch_data['reordered_targets']
        
            if rank == 0:
                schedule.step(args_split, kwargs_split, chunked_sg_ori_node_idxes, split_idx=split_idx['train'])
            elif rank == num_stages - 1:
                losses = []
                output = schedule.step(args_split, kwargs_split, chunked_sg_ori_node_idxes, target=labels_split, losses=losses, split_idx=split_idx['train']) 
                for i in range(len(losses)):
                    losses[i] = losses[i].item()
                loss = np.mean(losses)

            else:
                schedule.step(args_split, kwargs_split, chunked_sg_ori_node_idxes, split_idx=split_idx['train'])
            

            torch.nn.utils.clip_grad_norm_(stage.submod.parameters(), 1.0)
            optimizer.step()
            # pp_profile.step()
            
        t1 = time.time()
        epoch_time_list.append(t1 - t0)
        if epoch > 4:
            if rank == num_stages - 1:
                print(
                        "Epoch {:05d} | Loss {:.4f} | Avg Epoch Time {:.4f}s".format(
                            epoch, loss, np.mean(epoch_time_list[5:])
                        )
                    )
        else:
            if rank == num_stages - 1:
                print(
                        "Epoch {:05d} | Loss {:.4f} | Epoch Time {:.2f}s".format(
                            epoch, loss, t1 - t0
                        )
                    )
            
        # pp_profile.stop()
        # Validation
        if epoch % 5 == 0:
            results = evaluate(args, preprocessed_batches, data_processed.batch_nodes, split_idx, stage, schedule)
            if rank == num_stages - 1:
                print("Epoch {:05d} | Valid acc: {:.2f}% | Test acc: {:.2f}%".format(epoch, results[0], results[1]))
                loss_list.append(loss)
                val_acc_list.append(results[0])
                test_acc_list.append(results[1])

    timestamp1 = time.time()
    local_time = time.localtime(timestamp1)
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    ms = int((timestamp1 - int(timestamp1)) * 1000)
    formatted_time = f"{time_str}.{ms:03d}"
    print(f"{args.gnn_model} finish time: {formatted_time}, wall-time: {timestamp1-timestamp0:.3f}s")
        

    if rank == num_stages - 1:
        print("Best TestAcc: {:.4f}".format(max(test_acc_list)))

        if not os.path.exists(f'./exps/{args.dataset}'): 
            os.makedirs(f'./exps/{args.dataset}')
        np.save(f'./exps/{args.dataset}/{args.num_layers}{args.gnn_model}_{num_stages}pp_{data_processed.num_batches}b_{num_microbatches}mb_{args.lm_model}_loss', np.array(loss_list))
        np.save(f'./exps/{args.dataset}/{args.num_layers}{args.gnn_model}_{num_stages}pp_{data_processed.num_batches}b_{num_microbatches}mb_{args.lm_model}_val_acc', np.array(val_acc_list))
        np.save(f'./exps/{args.dataset}/{args.num_layers}{args.gnn_model}_{num_stages}pp_{data_processed.num_batches}b_{num_microbatches}mb_{args.lm_model}_test_acc', np.array(test_acc_list))

    dist.destroy_process_group()
