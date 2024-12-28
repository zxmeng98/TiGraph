import numpy as np
import torch
import dgl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial
import scipy.sparse as sp
import scipy
from torch_geometric.datasets import Planetoid, Amazon, Actor, CitationFull, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce
import os
import pickle as pkl
from ogb.nodeproppred import DglNodePropPredDataset, NodePropPredDataset


def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def eigenvector(L):
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.tensor(EigVec[:, 1:11], dtype = torch.float32)


def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def preprocess(graph):
    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def get_dataset(dataset_name):
    # print(f'Get dataset {dataset_name}...')

    dataset_dir = "/home/mzhang/data/"
    # if not os.path.exists(f'{dataset_dir}/{dataset_name}'): 
    #     os.makedirs(f'{dataset_dir}/{dataset_name}')

    # if os.path.exists(dataset_path):
    #     print(f'Already downloaded. Loading {dataset_name}...')
    #     data_x = torch.load(dataset_dir + dataset_name + '/x.pt')
    #     data_y = torch.load(dataset_dir + dataset_name + '/y.pt')
    #     adj = sp.load_npz(dataset_dir + dataset_name + '/adj.npz')
        
    #     normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/normalized_adj.npz')
    #     column_normalized_adj = sp.load_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz')
    # else: 
    if True:
        if dataset_name in ['cora', 'citeseer', 'pubmed']: 
            dataset = Planetoid(root=dataset_dir, name=dataset_name)       
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in ['dblp']:
            dataset = CitationFull(root=dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)
                
        elif dataset_name in ["CS", "Physics"]:
        # TODO: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Coauthor.html
            dataset = Coauthor(root=dataset_dir, name=dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in ["Photo"]:
            dataset = Amazon(root=dataset_dir, name=dataset_name)
            data = dataset[0]
            data_x = data.x
            data_y = data.y
            edge_index = data.edge_index
            
            adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)
        
        elif dataset_name in ['aminer']:
            adj = pkl.load(open(os.path.join(dataset_dir, dataset_name, "{}.adj.sp.pkl".format(dataset_name)), "rb"))
            data_x = pkl.load(
                open(os.path.join(dataset_dir, dataset_name, "{}.features.pkl".format(dataset_name)), "rb"))
            data_y = pkl.load(
                open(os.path.join(dataset_dir, dataset_name, "{}.labels.pkl".format(dataset_name)), "rb"))
            # random_state = np.random.RandomState(split_seed)
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))
            
        elif dataset_name in ['reddit']:
            adj = sp.load_npz(os.path.join(dataset_dir, dataset_name, '{}_adj.npz'.format(dataset_name)))
            data_x = np.load(os.path.join(dataset_dir, dataset_name, '{}_feat.npy'.format(dataset_name)))
            data_y = np.load(os.path.join(dataset_dir, dataset_name, '{}_labels.npy'.format(dataset_name)))
            # random_state = np.random.RandomState(split_seed)
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))

            
        elif dataset_name in ['Amazon2M']:
            adj = sp.load_npz(os.path.join(dataset_dir, dataset_name, '{}_adj.npz'.format(dataset_name)))
            data_x = np.load(os.path.join(dataset_dir, dataset_name, '{}_feat.npy'.format(dataset_name)))
            data_y = np.load(os.path.join(dataset_dir, dataset_name, '{}_labels.npy'.format(dataset_name)))
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))
            
        elif dataset_name in ['amazon']:
            dataset_dir = "/home/mzhang/data/"
            adj = sp.load_npz(os.path.join(dataset_dir, dataset_name, 'adj_full.npz'))
            data_x = np.load(os.path.join(dataset_dir, dataset_name, 'feats.npy'))
            data_y = np.load(os.path.join(dataset_dir, dataset_name, 'labels.npy'))
            data_x = torch.tensor(data_x, dtype=torch.float32)
            data_y = torch.tensor(data_y)
            data_y = torch.argmax(data_y, -1)
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

            row, col = adj.nonzero()
            row = torch.from_numpy(row).to(torch.long)
            col = torch.from_numpy(col).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            edge_index = coalesce(edge_index, num_nodes=data_x.size(0))
            
        elif dataset_name in ['pokec']:
            fulldata = scipy.io.loadmat(f'/home/mzhang/work/GTNodeLevel/dataset/pokec.mat')
            edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
            
            data_x = torch.tensor(fulldata['node_feat']).float()
            label = fulldata['label'].flatten()
            data_y = torch.tensor(label, dtype=torch.long)
            
            num_nodes = data_y.shape[0]
            adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                                        shape=(num_nodes, num_nodes), dtype=np.float32)
          
            
            normalized_adj = adj_normalize(adj)
            column_normalized_adj = column_normalize(adj)

        elif dataset_name in {"ogbn-papers100M"}:
            file_dir = '/home/mzhang/data/'
            ogb_dataset = NodePropPredDataset(name=dataset_name, root=file_dir)
            split_idx = ogb_dataset.get_idx_split()
            idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
            
            data_y = torch.as_tensor(ogb_dataset.labels).squeeze(1)
            # data_x = torch.as_tensor(ogb_dataset.graph['node_feat'])
            edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
            # num_nodes=ogb_dataset.graph['num_nodes']
            # adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            #                         shape=(num_nodes, num_nodes), dtype=np.float32)
            # normalized_adj = adj_normalize(adj)
            # column_normalized_adj = column_normalize(adj)
        
        elif dataset_name in ["ogbn-arxiv", "ogbn-products"]:
            ogb_dataset = DglNodePropPredDataset(name=dataset_name, root=dataset_dir)
            g, data_y = ogb_dataset[0]
            
            num_nodes = g.num_nodes()
            nodes_to_remove = torch.tensor([num_nodes-3, num_nodes-2, num_nodes-1])
            g.remove_nodes(nodes_to_remove) 
            data_y = torch.as_tensor(data_y).squeeze(1)
            data_y = data_y[:-3]
            # g = preprocess(g)
            
            split_idx = ogb_dataset.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            
            keep_in_train = ~torch.isin(train_idx, nodes_to_remove)
            train_idx = train_idx[keep_in_train]

            keep_in_valid = ~torch.isin(valid_idx, nodes_to_remove)
            valid_idx = valid_idx[keep_in_valid]

            keep_in_test = ~torch.isin(test_idx, nodes_to_remove)
            test_idx = test_idx[keep_in_test]
            split_idx = {'train': train_idx.to(torch.int32),
                'valid': valid_idx.to(torch.int32),
                'test': test_idx.to(torch.int32)}
            
            data_x = g.ndata["feat"]
            # num_nodes = ogb_dataset.graph['num_nodes']
            # adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            #                         shape=(num_nodes, num_nodes), dtype=np.float32)
            # normalized_adj = adj_normalize(adj)
            # column_normalized_adj = column_normalize(adj)

        # sp.save_npz(dataset_dir + dataset_name + '/adj.npz', adj)
        # sp.save_npz(dataset_dir + dataset_name + '/normalized_adj.npz', normalized_adj)
        # dataset_dir = './dataset/'
        # torch.save(data_x, dataset_dir + dataset_name + '/x.pt')
        # torch.save(data_y, dataset_dir + dataset_name + '/y.pt')
        # torch.save(edge_index, dataset_dir + dataset_name + '/edge_index.pt')
        # sp.save_npz(dataset_dir + dataset_name + '/column_normalized_adj.npz', column_normalized_adj)
        return g, split_idx, data_x, data_y
    

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
