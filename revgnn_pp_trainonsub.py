# Inductive train: training only sees graph structure including train nodes. Aim to train models on nodes of a training graph, and then generalize models to structure-unobserved nodes.
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
from utils.dataset import OGBNDataset, random_split_idx
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe
from pipelining.utils import partition_uniform


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
    print(f'stage={stage_index} layers={stop} - {start}')
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
    parser.add_argument('--dataset', type=str, default='pubmed',
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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.2)

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
    parser.add_argument('--mb_size', type=int, default=20,
                       help='Number of microbatches.')
    
    # model
    parser.add_argument('--model', type=str, default='revgnn',
                        help='gcn backbone [revgnn, resgnn]')
    parser.add_argument('--backbone', type=str, default='rev',
                        help='gcn backbone [deepergcn, weighttied, deq, rev]')
    parser.add_argument('--group', type=int, default=2,
                        help='num of groups for rev gnns')
    parser.add_argument('--num_layers', type=int, default=448,
                        help='the number of layers of the networks')
    parser.add_argument('--num_steps', type=int, default=3,
                        help='the number of steps of weight tied layers')
    parser.add_argument('--mlp_layers', type=int, default=2,
                        help='the number of layers of mlp in conv')
    parser.add_argument('--in_size', type=int, default=50,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--out_size', type=int, default=3,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--hidden_channels', type=int, default=224,
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
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    
    g = data[0]
    g = g.int()
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # Generate train data
    if masks is None:
        split_idx = random_split_idx(labels, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)
    else:
        train_idx = torch.nonzero(masks[0], as_tuple=True)[0]  
        val_idx = torch.nonzero(masks[1], as_tuple=True)[0]  
        test_idx = torch.nonzero(masks[2], as_tuple=True)[0]    
        split_idx = {'train': train_idx.to(torch.int32),
                'valid': val_idx.to(torch.int32),
                'test': test_idx.to(torch.int32)}
    if rank == 0:
        print('Dataset load successfully')
        print(f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}")
    
    features = features.to(device)
    lables = labels.to(device)
    labels_one_hot = F.one_hot(labels, num_classes=args.out_size).float() 
    labels_one_hot = labels_one_hot.to(device)

    # Create train subset
    train_g = dgl.node_subgraph(g, split_idx['train'])
    train_g = train_g.to(device)
    train_features = features[split_idx['train']]
    train_labels = labels_one_hot[split_idx['train']]

    valid_g = dgl.node_subgraph(g, split_idx['valid'])
    valid_g = valid_g.to(device)
    valid_features = features[split_idx['valid']]
    valid_labels = labels_one_hot[split_idx['valid']]

    test_g = dgl.node_subgraph(g, split_idx['test'])
    test_g = test_g.to(device)
    test_features = features[split_idx['test']]
    test_labels = labels_one_hot[split_idx['test']]

    num_microbatches = split_idx['train'].shape[0] // args.mb_size
    microbatch_idxes = torch.tensor_split(split_idx['train'], num_microbatches)
    microbatch_g = dgl.node_subgraph(g, microbatch_idxes[0].to(torch.int32))
    microbatch_g.ndata.clear()
    microbatch_g.edata.clear()
    u, v = microbatch_g.edges()
    mb_edges_ = torch.stack((u, v), dim=0).to(device)
    example_input_microbatch = (microbatch_g, features[microbatch_idxes[0]])

    # create GCN model
    args.in_size = features.shape[1]
    args.out_size = data.num_classes

    model = RevGCN(args)
    stage = manual_model_split(args, model, example_input_microbatch)

    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(stage.submod.parameters(), lr=args.lr)

    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=loss_fcn)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    if rank == 0:
        print("Training...")

    loss_list, val_acc_list = [], []
    # Training loop
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        stage.submod.train()
        if rank == 0:
            schedule.step(train_g, train_features)
        elif rank == num_stages - 1:
            losses = []
            output = schedule.step(train_g, target=train_labels, losses=losses)
            loss = loss_fcn(output, train_labels)
            loss_list.append(loss.item())

            for i in range(len(losses)):
                losses[i] = losses[i].item()
        else:
            schedule.step(g)

        torch.nn.utils.clip_grad_norm_(stage.submod.parameters(), 1.0)
        optimizer.step()

        # Validation
        if epoch % 5 == 0:
            num_microbatches = split_idx['valid'].shape[0] // args.mb_size
            valid_schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=loss_fcn)
            val_acc = evaluate(args, valid_g, valid_features, valid_labels, masks[1], stage, valid_schedule)
            if rank == num_stages - 1:  
                print(
                    "Epoch {:05d} | Loss {:.4f} | Val Accuracy {:.4f} ".format(
                        epoch, loss.item(), val_acc
                    )
                )
                loss_list.append(loss.item())
                val_acc_list.append(val_acc)
        

    # Test
    # if rank == 0:
    #     print("Testing...")
    # test_acc = evaluate(args, test_g, test_features, test_labels, masks[2], stage, schedule)
    # if rank == num_stages - 1:
    #     print("Test accuracy {:.4f}".format(test_acc))

        # if not os.path.exists(f'./exps/{args.dataset}'): 
        #     os.makedirs(f'./exps/{args.dataset}')
        # np.save(f'./exps/{args.dataset}/{args.model}_loss', np.array(loss_list))
        # np.save(f'./exps/{args.dataset}/{args.model}_val_acc', np.array(val_acc_list))

    dist.destroy_process_group()
