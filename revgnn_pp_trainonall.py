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
from GNNs.RevGNN.revgcn_pp import RevGCN
from utils.dataset import get_dataset
import torch.distributed as dist
from pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
from pipelining._utils import partition_uniform
from ogb.nodeproppred import DglNodePropPredDataset


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


def evaluate(args, g, features, labels, mask, stage, schedule):
    stage.submod.eval()
    if rank == 0:
        schedule.step(g, features)
        return None
    elif rank == num_stages - 1:
        losses = []
        output = schedule.step(g, target=labels, losses=losses)   
        eval_output = output
        _, indices = torch.max(eval_output, dim=1)
        _, eval_labels = torch.max(labels, dim=1)
        correct = torch.sum(indices == eval_labels)
        eval_acc = correct.item() * 1.0 / len(eval_labels)
        return eval_acc
    else:
        schedule.step(g)
        return None

        # print("Test accuracy {:.4f}".format(eval_acc))
    

if __name__ == "__main__":
    init_distributed()
        
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        help='dataset name (default: ogbn-proteins)')
    parser.add_argument('--cluster_number', type=int, default=10,
                        help='the number of sub-graphs for training')
    parser.add_argument('--valid_cluster_number', type=int, default=5,
                        help='the number of sub-graphs for evaluation')
    parser.add_argument('--aggr', type=str, default='add',
                        help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
    parser.add_argument('--nf_path', type=str, default='init_node_features_add.pt',
                        help='the file path of extracted node features saved.')
    
    # training & eval settings
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--num_evals', type=int, default=1,
                        help='The number of evaluation times')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.75)

    # pipeline pearallel 
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
    parser.add_argument('--bs', type=int, default=84670,
                       help='Number of microbatches.')
    parser.add_argument('--mb_size', type=int, default=42335,
                       help='Number of microbatches.')
    
    # model
    parser.add_argument('--model', type=str, default='revgnn',
                        help='gcn backbone [revgnn, resgnn]')
    parser.add_argument('--backbone', type=str, default='rev',
                        help='gcn backbone [deepergcn, weighttied, deq, rev]')
    parser.add_argument('--group', type=int, default=2,
                        help='num of groups for rev gnns')
    parser.add_argument('--num_layers', type=int, default=112,
                        help='the number of layers of the networks')
    parser.add_argument('--num_steps', type=int, default=3,
                        help='the number of steps of weight tied layers')
    parser.add_argument('--mlp_layers', type=int, default=2,
                        help='the number of layers of mlp in conv')
    parser.add_argument('--in_size', type=int, default=50,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--out_size', type=int, default=3,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--block', default='res', type=str,
                        help='graph backbone block type {res+, res, dense, plain}')
    parser.add_argument('--conv', type=str, default='gen',
                        help='the type of GCNs')
    parser.add_argument('--gcn_aggr', type=str, default='max',
                        help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
    parser.add_argument('--norm', type=str, default='layer',
                        help='the type of normalization layer')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='the number of prediction tasks')
    
    # learnable parameters
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
    # save model
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',
                        help='the directory used to save models')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    # load pre-trained model
    parser.add_argument('--model_load_path', type=str, default='ogbn_proteins_pretrained_model.pth',
                        help='the path of pre-trained model')
    # deq
    parser.add_argument('--inject_input', action='store_true')
    parser.add_argument('--pretrain_epochs', type=int, default=100,
                        help='number of epochs to pretrain (default: 100)')
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    elif args.dataset == "ogbn-arxiv":
        data = DglNodePropPredDataset(name=args.dataset, root="/home/mzhang/data/")
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    
    masks = None
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        g = data[0]
        g = g.int()
        # g.remove_nodes(np.arange(5)) 
        features = g.ndata["feat"]
        labels = g.ndata["label"]
        masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    elif args.dataset == "ogbn-arxiv":
        g, split_idx, features, labels = get_dataset(args.dataset)
    args.in_size = features.shape[1]
    args.out_size = data.num_classes
    
    # Generate train data
    if masks is not None:
        train_idx = torch.nonzero(masks[0], as_tuple=True)[0]  
        val_idx = torch.nonzero(masks[1], as_tuple=True)[0]  
        test_idx = torch.nonzero(masks[2], as_tuple=True)[0]    
        split_idx = {'train': train_idx.to(torch.int32),
                'valid': val_idx.to(torch.int32),
                'test': test_idx.to(torch.int32)}
    if rank == 0:
        print('Dataset load successfully')
        print(f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}")
    
    all_idx = torch.randperm(features.shape[0])
    g.ndata['random_idx'] = all_idx
    num_batches = all_idx.shape[0] // args.bs # TODO 没有考虑到不整除的情况
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
        packed_batch.append((g, features, labels))

    # Create GCN model
    model = RevGCN(args)
    stage = manual_model_split(args, model, example_input_microbatch)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(stage.submod.parameters(), lr=args.lr)
    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=loss_fcn)

    # Convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # Model training
    if rank == 0:
        print("Training...")

    loss_list, val_acc_list = [], []
    # Training loop
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        stage.submod.train()
        for i in range(num_batches):
            g_i, features_i, labels_i = packed_batch[i]
            g_i, features_i, labels_i = g_i.to(device), features_i.to(device), labels_i.to(device)
            if rank == 0:
                schedule.step(g_i, features_i, split_idx=split_idx['train'], batch_idx=i)
            elif rank == num_stages - 1:
                losses = []
                output = schedule.step(g_i, target=labels_i, losses=losses, split_idx=split_idx['train'], batch_idx=i) 
                for i in range(len(losses)):
                    losses[i] = losses[i].item()
                loss = np.mean(losses)
                loss_list.append(loss)
                print(
                    "Epoch {:05d} | Loss {:.4f} ".format(
                        epoch, loss
                    )
                )

            else:
                schedule.step(g_i, split_idx=split_idx['train'], batch_idx=i)

            torch.nn.utils.clip_grad_norm_(stage.submod.parameters(), 1.0)
            optimizer.step()

        # # Validation
        # if epoch % 5 == 0:
        #     num_microbatches = split_idx['valid'].shape[0] // args.mb_size
        #     valid_schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=loss_fcn, masks=masks)
        #     val_acc = evaluate(args, valid_g, valid_features, valid_labels, masks[1], stage, valid_schedule)
        #     if rank == num_stages - 1:  
        #         print(
        #             "Epoch {:05d} | Loss {:.4f} | Val Accuracy {:.4f} ".format(
        #                 epoch, loss.item(), val_acc
        #             )
        #         )
        #         loss_list.append(loss.item())
        #         val_acc_list.append(val_acc)
        

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
