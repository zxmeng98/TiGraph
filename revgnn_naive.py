import argparse

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
from GNNs.RevGNN.revgcn import RevGCN
from GNNs.resgnn import DeeperGCN
from GNNs.revgat import RevGAT
from utils.dataset import get_dataset, adjust_dataset


def evaluate(g, features, labels, split_idx, model, device):
    sg_nodes_idx = g.nodes().to(device)
    u, v = g.edges()
    sg_edges_ = torch.stack((u, v), dim=0).to(device)
    # labels_one_hot = F.one_hot(labels, num_classes=3).float() 

    model.eval()
    with torch.no_grad():
        logits = model(g.to(device), features)
        logits = logits[split_idx]
        labels = labels[split_idx]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 100.0 / len(labels)


def train(g, features, labels, split_idx, model, device):
    # define train/val samples, loss function and optimizer
    train_idx, valid_idx = split_idx['train'], split_idx['valid']
    loss_fcn = nn.CrossEntropyLoss()
    if args.model == 'revgnn' or args.model == 'resgnn':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'revgat':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0)


    # features = features.chunk(4)[0]

    loss_list, val_acc_list, test_acc_list, epoch_time_list = [], [], [], []
    # training loop
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_idx], labels[train_idx])
        # loss = loss_fcn(logits, labels_one_hot)
        optimizer.zero_grad()
        loss.backward()
        if args.model == 'revgnn':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t1 = time.time()
        epoch_time_list.append(t1 - t0)
        acc = evaluate(g, features, labels, valid_idx, model, device)
        test_acc = evaluate(g, features, labels, split_idx['test'], model, device)
        # print(
        #     "Epoch {:05d} | Loss {:.4f} ".format(
        #         epoch, loss.item()
        #     )
        # )
        print(
            "Epoch {:05d} | Loss {:.4f} | Val Acc {:.2f}% | Test Acc {:.2f}% | Epoch Time {:.2f}s".format(
                epoch, loss.item(), acc, test_acc, t1 - t0
            )
        )
        
        
        loss_list.append(loss.item())
        val_acc_list.append(acc)
        test_acc_list.append(test_acc)

    print("Test Acc {:.4f}".format(max(test_acc_list)))
    if not os.path.exists(f'./exps/{args.dataset}'): 
        os.makedirs(f'./exps/{args.dataset}')
    np.save(f'./exps/{args.dataset}/{args.model}_loss_{args.num_layers}layers', np.array(loss_list))
    np.save(f'./exps/{args.dataset}/{args.model}_val_acc_{args.num_layers}layers', np.array(val_acc_list))
    np.save(f'./exps/{args.dataset}/{args.model}_test_acc_{args.num_layers}layers', np.array(test_acc_list))


if __name__ == "__main__":
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        help='dataset name (default: ogbn-proteins)')
    parser.add_argument('--cluster_number', type=int, default=10,
                        help='the number of sub-graphs for training')
    parser.add_argument('--valid_cluster_number', type=int, default=5,
                        help='the number of sub-graphs for evaluation')
    parser.add_argument('--aggr', type=str, default='max',
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
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.75)
    parser.add_argument('--bs', type=int, default=169343,
                       help='Number of microbatches.')
    # model
    parser.add_argument('--model', type=str, default='revgnn',
                        help='gcn backbone [revgnn, resgnn, revgat]')
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
    parser.add_argument('--hidden_channels', type=int, default=224,
                        help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--conv', type=str, default='gen',
                        help='the type of GCNs')
    parser.add_argument('--gcn_aggr', type=str, default='max',
                        help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
    parser.add_argument('--norm', type=str, default='layer',
                        help='the type of normalization layer')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='the number of prediction tasks')
    parser.add_argument('--block', default='res+', type=str,
                        help='graph backbone block type {res+, res, dense, plain}')
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
    torch.cuda.set_device(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load and preprocess dataset
    g, split_idx, features, labels, num_classes = get_dataset(args.dataset)
    LM_emb_path = f"./lm_workloads/prt_lm/{args.dataset}/microsoft/deberta-base-seed0.emb"
    if os.path.exists(LM_emb_path):
        print("Loading trained LM features (title and abstract) ...")
        print(f"LM_emb_path: {LM_emb_path}")
        features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                        dtype=np.float16,
                        shape=(g.num_nodes(), 768)))
        ).to(torch.float32)
        g.ndata['feat'] = features
    g, split_idx, features, labels = adjust_dataset(args, g, split_idx, features, labels)
    if args.model == 'revgat':
        g = dgl.to_bidirected(g)
        g = g.remove_self_loop().add_self_loop()
        g.create_formats_()

    g = g.to(device)
    features = features.to(device)
    labels = labels.to(device)
    for idx in split_idx.values():
        idx.to(device)

    # create GCN model
    args.in_size = features.shape[1]
    args.out_size = num_classes

    if args.model == 'revgat':
        model = RevGAT(
                        args.in_size,
                        args.out_size,
                        n_hidden=args.hidden_channels,
                        n_layers=args.num_layers,
                        n_heads=3,
                        activation=F.relu,
                        dropout=args.dropout,
                        input_drop=0.25,
                        attn_drop=0.0,
                        edge_drop=0.3,
                        use_attn_dst=False,
                        use_symmetric_norm=True,
                        number_of_edges=g.num_edges(),
                        ).to(device)
    elif args.model == 'revgnn':
        model = RevGCN(args).to(device)
    elif args.model == 'resgnn':
        model = DeeperGCN(args).to(device)
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nNumber of parameters: {trainable_params}")

    # for name, param in model.named_parameters():
    #     print(name, param)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    train(g, features, labels, split_idx, model, device)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, split_idx['test'], model, device)
    print("Test accuracy {:.4f}".format(acc))
