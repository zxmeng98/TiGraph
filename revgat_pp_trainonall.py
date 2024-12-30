import argparse

import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import random
import numpy as np
import os
from GNNs.revgat_pp import RevGAT
from utils.dataset import get_dataset, adjust_dataset
import torch.distributed as dist
from pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
from pipelining._utils import partition_uniform
from ogb.nodeproppred import DglNodePropPredDataset
from utils.dataset import intersection


# global rank, device, pp_group, stage_index, num_stages
# def init_distributed():
#    global rank, device, pp_group, stage_index, num_stages
#    rank = int(os.environ["LOCAL_RANK"])
#    world_size = int(os.environ["WORLD_SIZE"])
#    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
#    dist.init_process_group("nccl")

#    # This group can be a sub-group in the N-D parallel case
#    pp_group = dist.new_group()
#    stage_index = rank
#    num_stages = world_size
   
global rank, device, pp_group, stage_index, num_stages
def init_distributed():
    """Initialize torch.distributed and core model parallel."""
    global rank, device, pp_group, stage_index, num_stages
    device_count = torch.cuda.device_count()
    assert device_count != 0, 'expected GPU number > 0.'
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if rank == 0:
            print('> initializing torch distributed ...', flush=True)

        # Manually set the device ids.
        if device_count > 0:
            device = rank % device_count
            torch.cuda.set_device(device) # only do so when device_count > 0
        # Call the init process
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size, rank=rank,
        )
    device = f'cuda:{torch.cuda.current_device()}' 
    pp_group = dist.new_group()
    stage_index = rank
    num_stages = world_size


def manual_model_split(args, model, example_input_microbatch) -> PipelineStage:
    parts = partition_uniform(args.num_layers, num_stages)
    start = parts[stage_index]
    stop = parts[stage_index + 1]
    print(f'stage={stage_index} layers={start} - {stop-1}')
    if stage_index == 0:
        # prepare the first stage model
        model.first_stage = True
        model.last_stage = False
        model.convs = model.convs[start:stop]
        model.perms = model.perms[start:stop] 
        model.dp_last = None
        model.bias_last = None
        # print(f"after:{sum(p.numel() for p in model.parameters())}")
        # exit()
        stage_input_microbatch = example_input_microbatch

    elif stage_index == num_stages - 1:
        # prepare the second stage model
        model.first_stage = False
        model.last_stage = True
        model.convs = model.convs[start:stop]
        model.perms = model.perms[start:stop] 

        feature_input_microbatch = torch.randn(example_input_microbatch[1].shape[0], args.hidden_channels * args.num_heads)
        stage_input_microbatch = (example_input_microbatch[0], feature_input_microbatch)

    else:
        model.first_stage = False
        model.last_stage = False
        model.convs = model.convs[start:stop]
        model.perms = model.perms[start:stop] 
        model.dp_last = None
        model.bias_last = None
        feature_input_microbatch = torch.randn(example_input_microbatch[1].shape[0], args.hidden_channels * args.num_heads)
        stage_input_microbatch = (example_input_microbatch[0], feature_input_microbatch)
        
    stage = PipelineStage(
      model,
      stage_index,
      num_stages,
      device,
      input_args=stage_input_microbatch,
   )
    return stage


def evaluate(args, packed_batch, batch_idxes, split_idx, stage, schedule):
    valid_output, test_output = [], []
    valid_labels, test_labels = [], []
    stage.submod.eval()
    for i in range(len(batch_idxes)):
        g_i, features_i, labels_i = packed_batch[i]
        g_i, features_i, labels_i = g_i.to(device), features_i.to(device), labels_i.to(device)
        batch_idx = batch_idxes[i].tolist()
        if rank == 0:
            schedule.step(g_i, features_i, split_idx=split_idx['valid'])
        elif rank == num_stages - 1:
            output = schedule.step(g_i, target=labels_i, split_idx=split_idx['valid'])   
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
            schedule.step(g_i, split_idx=split_idx['valid'])
    
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
        print("Valid acc: {:.2f}%. Test acc: {:.2f}%".format(valid_acc, test_acc))
        return (valid_acc, test_acc)
        
    

if __name__ == "__main__":
    init_distributed()
        
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        help='dataset name (default: ogbn-proteins)')
    
    # Training & eval settings
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.75)
    parser.add_argument('--bs', type=int, default=169343,
                       help='Number of microbatches.')
    parser.add_argument('--mb_size', type=int, default=169343,
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

    # Model
    parser.add_argument('--model', type=str, default='revgnn',
                        help='gcn backbone [revgnn, resgnn]')
    parser.add_argument('--backbone', type=str, default='rev',
                        help='gcn backbone [deepergcn, weighttied, deq, rev]')
    parser.add_argument('--group', type=int, default=2,
                        help='num of groups for rev gnns')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='the number of layers of the networks')
    parser.add_argument('--mlp_layers', type=int, default=2,
                        help='the number of layers of mlp in conv')
    parser.add_argument('--in_size', type=int, default=50,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--out_size', type=int, default=3,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--gcn_aggr', type=str, default='max',
                        help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
    parser.add_argument('--norm', type=str, default='layer',
                        help='the type of normalization layer')
    parser.add_argument("--num_heads", type=int, default=3, help="number of heads")
    
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

    # Load and preprocess dataset
    g, split_idx, features, labels, num_classes = get_dataset(args.dataset)
    # g, split_idx, features, labels = adjust_dataset(args, g, split_idx, features, labels)
    g = dgl.to_bidirected(g)
    g = g.remove_self_loop().add_self_loop()
    g.create_formats_()

    # TODO: 
    # all_idx = torch.randperm(features.shape[0])
    all_idx = torch.arange(features.shape[0])
    g.ndata['random_idx'] = all_idx

    num_batches = all_idx.shape[0] // args.bs 
    args.in_size = features.shape[1]
    args.out_size = num_classes
    
    if rank == 0:
        print(f"{args.dataset} load successfully")
        print(f"Total nodes: {g.num_nodes()}, Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}")

    batch_idxes = torch.tensor_split(all_idx, num_batches)
    
    if num_batches > 1:
        batch_g = dgl.node_subgraph(g, batch_idxes[0])
    else:
        batch_g = g
    
    num_microbatches = batch_idxes[0].shape[0] // args.mb_size
    batch_local_idxes = torch.arange(batch_g.num_nodes())
    microbatch_idxes = torch.tensor_split(batch_local_idxes, num_microbatches)
    microbatch_g = dgl.node_subgraph(batch_g, microbatch_idxes[0])
    
    microbatch_g.ndata.clear()
    microbatch_g.edata.clear()
    example_input_microbatch = (microbatch_g, features[microbatch_idxes[0]])

    if rank == 0:
        print(f"Num of train batches: {num_batches}, num of train microbatches: {num_microbatches}, microbatch size: {args.mb_size}")
    
    # Pack batch graph trained in each iter beforehand
    packed_batch = []
    if num_batches > 1:
        for i in range(num_batches):
            idx_i = batch_idxes[i]
            g_i = dgl.node_subgraph(g, idx_i)
            features_i = features[idx_i]
            labels_i = labels[idx_i]
            packed_batch.append((g_i, features_i, labels_i))
    elif num_batches == 1:
        features = features[batch_idxes[0]]
        labels = labels[batch_idxes[0]]
        packed_batch.append((g, features, labels))
        labels = labels.to(device)

    # Create GCN model
    model = RevGAT(
                    args.in_size,
                    args.out_size,
                    n_hidden=args.hidden_channels,
                    n_layers=args.num_layers,
                    n_heads=args.num_heads,
                    activation=F.relu,
                    dropout=args.dropout,
                    input_drop=0.25,
                    attn_drop=0.0,
                    edge_drop=0.0,
                    use_attn_dst=False,
                    use_symmetric_norm=True,
                    number_of_edges=packed_batch[0][0].num_edges(),
                    )
    stage = manual_model_split(args, model, example_input_microbatch)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(stage.submod.parameters(), lr=args.lr, weight_decay=0)
    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=loss_fcn)

    # Convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # Model training
    if rank == 0:
        print("Training...")

    # Training loop
    loss_list, val_acc_list = [], [] 
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        stage.submod.train()
        for i in range(num_batches):
            g_i, features_i, labels_i = packed_batch[i]
            g_i, features_i, labels_i = g_i.to(device), features_i.to(device), labels_i.to(device)
            if rank == 0:
                schedule.step(g_i, features_i, split_idx=split_idx['train'])
            elif rank == num_stages - 1:
                losses = []
                output = schedule.step(g_i, target=labels_i, losses=losses, split_idx=split_idx['train']) 
                for i in range(len(losses)):
                    losses[i] = losses[i].item()
                loss = np.mean(losses)
                loss_list.append(loss)
                test_output = output[split_idx['test']]
                test_labels = labels[split_idx['test']]
                _, indices = torch.max(test_output, dim=1)
                correct = torch.sum(indices == test_labels)
                acc = correct.item() * 100.0 / len(test_labels)
                # print("Test accuracy {:.2f}".format(acc))

            else:
                schedule.step(g_i, split_idx=split_idx['train'])

            # torch.nn.utils.clip_grad_norm_(stage.submod.parameters(), 1.0)
            optimizer.step()
        
        if rank == num_stages - 1:
            print(
                    "Epoch {:05d} | Loss {:.4f} | test acc {:.2f}".format(
                        epoch, loss, acc
                    )
                )

        # Validation
        if epoch % 5 == 0:
            results = evaluate(args, packed_batch, batch_idxes, split_idx, stage, schedule)
            # if rank == num_stages - 1:  
            #     print(
            #         "Epoch {:05d} | Loss {:.4f} | Val Accuracy {:.4f} ".format(
            #             epoch, loss.item(), val_acc
            #         )
            #     )
            #     loss_list.append(loss.item())
            #     val_acc_list.append(val_acc)
        

    # Test
    # if rank == 0:
    #     print("Testing...")
    # test_acc = evaluate(args, test_g, test_features, test_labels, masks[2], stage, schedule)
    if rank == num_stages - 1:
        # print("Test accuracy {:.4f}".format(test_acc))

        if not os.path.exists(f'./exps/{args.dataset}'): 
            os.makedirs(f'./exps/{args.dataset}')
        # np.save(f'./exps/{args.dataset}/{args.model}_pp_loss-19712', np.array(loss_list))
        # np.save(f'./exps/{args.dataset}/{args.model}_val_acc', np.array(val_acc_list))

    dist.destroy_process_group()
