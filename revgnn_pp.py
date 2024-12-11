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
from GNNs.RevGNN.revgcn import RevGCN
from utils.dataset import OGBNDataset
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe


global rank, device, pp_group, stage_index, num_stages
def init_distributed():
   global rank, device, pp_group, stage_index, num_stages
   rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
   dist.init_process_group()

   # This group can be a sub-group in the N-D parallel case
   pp_group = dist.new_group()
   stage_index = rank
   num_stages = world_size


def manual_model_split(model, example_input_microbatch) -> PipelineStage:
    if stage_index == 0:
        # prepare the first stage model
        # del model.layers["2"]
        model.dropout = None
        model.layers["2"] = None
        stage_input_microbatch = example_input_microbatch

    elif stage_index == 1:
        # prepare the second stage model
        # del model.layers["1"]
        model.layers["1"] = None
        # NOTE: 不管哪个stage，self.inputs的格式都要按照最开始进入forward的args来，但是h是从中间状态开始。所以这里stage 1不能只传入h，要和forward格式一样传入(g, h)
        feature_input_microbatch = torch.randn(example_input_microbatch[1].shape[0], 16)
        stage_input_microbatch = (example_input_microbatch[0], feature_input_microbatch)
        
    # print(f"{rank}, {model}")
    # exit(0)

    stage = PipelineStage(
      model,
      stage_index,
      num_stages,
      device,
      input_args=stage_input_microbatch,
   )
    return stage


def evaluate(g, features, labels, mask, model, device):
    sg_nodes_idx = g.nodes().to(device)
    u, v = g.edges()
    sg_edges_ = torch.stack((u, v), dim=0).to(device)
    # labels_one_hot = F.one_hot(labels, num_classes=3).float() 

    model.eval()
    with torch.no_grad():
        logits = model(features, sg_edges_)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model, device):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sg_nodes_idx = g.nodes().to(device)
    u, v = g.edges()
    sg_edges_ = torch.stack((u, v), dim=0).to(device)
    labels_one_hot = F.one_hot(labels, num_classes=3).float() 

    # features = features.chunk(4)[0]

    loss_list, val_acc_list = [], []
    # training loop
    for epoch in range(50):
        model.train()
        logits = model(features, sg_edges_)
        # loss = loss_fcn(logits[train_mask], labels[train_mask])
        loss = loss_fcn(logits, labels_one_hot)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model, device)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} ".format(
        #         epoch, loss.item()
        #     )
        # )
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        # loss_list.append(loss.item())
        # val_acc_list.append(acc)
        # dataset = 'pubmed'
        # if not os.path.exists(f'./exps/{dataset}'): 
        #     os.makedirs(f'./exps/{dataset}')
        # np.save('./exps/' + dataset + '/sepfile_gcn_loss', np.array(loss_list))
        # np.save('./exps/' + dataset + '/sepfile_gcn_val_acc', np.array(val_acc_list))


if __name__ == "__main__":
    init_distributed()
    num_microbatches = 4

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
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
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--num_evals', type=int, default=1,
                        help='The number of evaluation times')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.2)
    # model
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
    parser.add_argument('--block', default='plain', type=str,
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
    args = parser.parse_args()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    all_idx = torch.arange(features.shape[0])
    microbatch_idxes = torch.tensor_split(all_idx, num_microbatches)
    microbatch_g = dgl.node_subgraph(g, microbatch_idxes[0].to(torch.int32))
    microbatch_g.ndata.clear()
    microbatch_g.edata.clear()
    u, v = microbatch_g.edges()
    sg_edges_ = torch.stack((u, v), dim=0).to(device)
    example_input_microbatch = (sg_edges_, features[microbatch_idxes[0]])

    # create GCN model
    args.in_size = features.shape[1]
    args.out_size = data.num_classes

    model = RevGCN(args)
    stage = manual_model_split(model, example_input_microbatch)

    # for name, param in model.named_parameters():
    #     print(name, param)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    train(g, features, labels, masks, model, device)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model, device)
    print("Test accuracy {:.4f}".format(acc))
