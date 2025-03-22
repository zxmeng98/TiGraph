import os
import sys
import random
import time
import pickle as pkl

import numpy as np
import scipy
import scipy.sparse as sp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

import dgl
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from torch_geometric.datasets import Amazon, CitationFull, Coauthor
from torch_geometric.utils import coalesce
import torch_geometric.transforms as T
from ogb.nodeproppred import DglNodePropPredDataset, NodePropPredDataset
import contextlib

from pipelining.initialize import (
    get_pipeline_parallel_rank,
)


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
    dataset_dir = "/home/mzhang/data/"
    if dataset_name in ['cora', 'citeseer', 'pubmed']: 
        transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
        if dataset_name == "cora":
            data = CoraGraphDataset(transform=transform)
        elif dataset_name == "citeseer":
            data = CiteseerGraphDataset(transform=transform)
        elif dataset_name == "pubmed":
            data = PubmedGraphDataset(transform=transform)
        g = data[0]
        g = g.int()
        data_x = g.ndata["feat"]
        data_y = g.ndata["label"]
        num_classes = data.num_classes
        masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
        train_idx = torch.nonzero(masks[0], as_tuple=True)[0]  
        val_idx = torch.nonzero(masks[1], as_tuple=True)[0]  
        test_idx = torch.nonzero(masks[2], as_tuple=True)[0]    
        split_idx = {'train': train_idx.to(torch.int32),
                'valid': val_idx.to(torch.int32),
                'test': test_idx.to(torch.int32)}

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
        data_y = torch.as_tensor(data_y).squeeze(1)
        split_idx = ogb_dataset.get_idx_split()
        data_x = torch.as_tensor(g.ndata["feat"])
        num_classes = ogb_dataset.num_classes
    elif dataset_name in ["ogbn-proteins"]:
        from torch_scatter import scatter
        ogb_dataset = DglNodePropPredDataset(name=dataset_name, root=dataset_dir)
        g, data_y = ogb_dataset[0]
        if data_y.shape[1] > 1:
            data_y = torch.argmax(data_y, dim=1)
        # data_y = torch.as_tensor(data_y).squeeze(1)
        split_idx = ogb_dataset.get_idx_split()
        num_classes = ogb_dataset.num_tasks
        u, v = g.edges()
        edge_index = torch.stack([u, v], dim=0)
        edge_attr = g.edata["feat"] 
        data_x = scatter(edge_attr,
                                        edge_index[0],
                                        dim=0,
                                        dim_size=g.num_nodes(),
                                        reduce='add')
        g.ndata["feat"] = data_x

    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))
        
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
    return g, split_idx, data_x, data_y, num_classes
    

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def adjust_dataset(args, g, split_idx, features, labels):
    rest_nums = features.shape[0] % args.bs

    if rest_nums == 0: 
        return g, split_idx, features, labels
    else:
        nodes_to_remove = torch.Tensor(range(features.shape[0] - rest_nums, features.shape[0])).long()
        g.remove_nodes(nodes_to_remove) 
        labels = labels[:features.shape[0] - rest_nums]
        
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
        return g, split_idx, data_x, labels


class DataProcess():
    def __init__(self, args, g, split_idx, features, labels):
        self.args = args
        self.dataset_name = args.dataset
        self.g = g
        self.split_idx = split_idx
        self.features = features
        self.labels = labels
        self.gnn_model_name = args.gnn_model
        self.lm_model_name = args.lm_model

        self.g, self.split_idx, self.features, self.labels = self.remove_nodes_cant_batch(g, split_idx, features, labels)
        self.num_nodes = self.features.shape[0]
        self.num_batches = self.num_nodes // args.bs 
        # all_nodes = torch.randperm(features.shape[0])
        all_nodes = torch.arange(self.num_nodes)
        self.g.ndata['random_idx'] = all_nodes
        self.batch_nodes = torch.tensor_split(all_nodes, self.num_batches)


    def remove_nodes_cant_batch(self, g, split_idx, features, labels):
        """
        Remove the rest nodes that are not enough to form a batch
        """
        rest_nums = features.shape[0] % self.args.bs
        # rest_nums = 2

        if rest_nums == 0: 
            return g, split_idx, features, labels
        else:
            nodes_to_remove = torch.Tensor(range(features.shape[0] - rest_nums, features.shape[0])).long()
            g.remove_nodes(nodes_to_remove) 
            labels = labels[:features.shape[0] - rest_nums]
            
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
            return g, split_idx, data_x, labels


    def pack_batch(self, features):
        """
        Pack batch graph trained in each iter beforehand
        """
        packed_batch = []
        if self.num_batches > 1:
            for i in range(self.num_batches):
                idx_i = self.batch_nodes[i]
                g_i = dgl.node_subgraph(self.g, idx_i)
                features_i = features[idx_i]
                labels_i = self.labels[idx_i]
                packed_batch.append((g_i, features_i, labels_i))
        elif self.num_batches == 1:
            features_i = features[self.batch_nodes[0]]
            labels_i = self.labels[self.batch_nodes[0]]
            packed_batch.append((self.g, features_i, labels_i))
        return packed_batch


    def check_load_from_lm(self, feature_type, last_written_rows):
        if feature_type == 'TA':     
            # LM_emb_path = f"prt_lm/{self.dataset_name}/{self.lm_model_name}.emb"
            LM_emb_path = f"./lm_workloads/prt_lm/ogbn-arxiv/microsoft/deberta-base-seed0.emb"
            if os.path.exists(LM_emb_path):
                if get_pipeline_parallel_rank() == 0:
                    print("Loading trained LM features (title and abstract) ...")
                    print(f"LM_emb_path: {LM_emb_path}")
                features = torch.from_numpy(np.array(
                        np.memmap(LM_emb_path, mode='r',
                                dtype=np.float16,
                                shape=(self.num_nodes, 768)))
                ).to(torch.float32)
                # load_lm_emb, last_written_rows = self.has_written_quarter(features, 1000, last_written_rows)
                # if load_lm_emb:
                #     features = self.override_null_with_gold(features)
                packed_data = self.pack_batch(features)
                return packed_data, last_written_rows
                # else:
                #     return None, last_written_rows

            else:
                print(f"LM embeddings not ready. Still use gold features.")
                return None, last_written_rows
        
    def override_null_with_gold(self, lm_emb):
        """
        Override the null embeddings in zeros with gold features
        """
        zero_rows = torch.where(torch.all(lm_emb == 0, dim=1))[0]
        if len(zero_rows) > 0:
            if get_pipeline_parallel_rank() == 0:
                print(f"Null embeddings found, override with gold embeddings.")
            lm_emb[zero_rows] = self.features[zero_rows]
            
        return lm_emb


    def has_written_quarter(self, emb, load_interval, last_written_rows):
        """
        Check whether the LM has written a quarter of embeddings
        """
        written_rows = torch.count_nonzero(emb[:, 0]).item()  # 假设数据第一列总是非零的
        if written_rows - last_written_rows >= load_interval:
            return True, written_rows
        return False, last_written_rows


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def create_pairs(N, M, off_N, off_M):
    """Create a list of pairs (a, b) where a ranges from 1 to N and b ranges from 1 to M."""
    return [(a, b) for a in range(off_N, off_N + N) for b in range(off_M, off_M + M)]


def generate_new_edges_optimized(edge_index, k, partition_ids, p, blocksize, new_id_mapping):
    # edge_counts = np.zeros((k, k), dtype=np.int64)
    new_edges = []
    src, dst = edge_index
    edge_partiton = (partition_ids.numpy()[src.numpy()], partition_ids.numpy()[dst.numpy()])
    edge_index_np = (src.numpy(), dst.numpy())
    # Combining the first and second rows of a for sorting
    combined_a = list(zip(edge_partiton[0], edge_partiton[1]))
    # Sorting combined_a and also sorting b based on the sorting of combined_a
    sorted_combined_a, sorted_b = zip(*sorted(zip(combined_a, zip(*edge_index_np))))
    # Unzipping the sorted lists
    sorted_a1, sorted_a2 = zip(*sorted_combined_a)
    sorted_b1, sorted_b2 = zip(*sorted_b)

    edge_count_list = np.zeros(k*k, dtype=np.int64)
    change_count = 0
    for i in range(len(sorted_a1)-1):
        if (sorted_a1[i] == sorted_a1[i+1]) and (sorted_a2[i] == sorted_a2[i+1]):
            edge_count_list[change_count] += 1
        else:
            edge_count_list[change_count] += 1
            change_count += 1
    edge_count_list[change_count] += 1
    cumulative_sum = [sum(edge_count_list[:i]) for i in range(len(edge_count_list) + 1)]

    # # Converting the sorted tuples back to lists
    # sorted_a_new = (list(sorted_a1), list(sorted_a2))
    # sorted_b_new = (list(sorted_b1), list(sorted_b2))
    
    # for src, dst in zip(*edge_index):
    #     src_part = partition_ids[src]
    #     dst_part = partition_ids[dst]
    #     edge_counts[src_part, dst_part] += 1

    node_counts = np.zeros(k, dtype=int)
    total_node_offset = np.zeros(k, dtype=int)
    for pid in partition_ids:
        node_counts[pid] += 1

    for i in range(k-1):
        total_node_offset[i+1] = total_node_offset[i] + node_counts[i]
    # print(total_node_offset)
    # node_counts = np.array([np.sum(partition_ids == i) for i in range(k)])
    # print(node_counts)
    keep_cnt = 0
    for i in range(k):
        for j in range(k):
            sparsity = edge_count_list[i*k+j] / (node_counts[i] * node_counts[j])
            # if i == j:
            # print(f"chunk sparsity: {sparsity:.8f}, {sparsity/8.7988461e-05}")

            if sparsity > p:
                raw_edge_src = sorted_b1[cumulative_sum[i*k+j]:cumulative_sum[i*k+j+1]]
                raw_edge_dst = sorted_b2[cumulative_sum[i*k+j]:cumulative_sum[i*k+j+1]]
                # print(len(raw_edge_src))
                for jj in range(len(raw_edge_src)):
                    new_edges.append((new_id_mapping[raw_edge_src[jj]], new_id_mapping[raw_edge_dst[jj]]))
                # for src, dst in zip(*edge_index):
                #     if partition_ids[src] == i and partition_ids[dst] == j:
                #         new_edges.append((new_id_mapping[src], new_id_mapping[dst]))
                #         # new_edges.append((src, dst))
            else:
                # mask = torch.zeros(node_counts[i], node_counts[j])
                num_elements_per_block = blocksize * blocksize

                total_elements = node_counts[i] * node_counts[j]
                num_nonzero_blocks = int(sparsity * total_elements / num_elements_per_block) + 1
                np.random.seed(keep_cnt)
                if (node_counts[i] >= blocksize) and (node_counts[j] >= blocksize):
                    for ii in range(num_nonzero_blocks):
                        block_row = (np.random.randint(0, node_counts[i] // blocksize)) #% (node_counts[i] // blocksize)
                        # print(ii, block_row)
                        block_col = (np.random.randint(0, node_counts[j] // blocksize)) #% (node_counts[j] // blocksize)
                        # print(ii, keep_cnt, block_row, block_col)
                        keep_cnt += 1
                        random_edge = create_pairs(blocksize, blocksize, (block_row * blocksize + total_node_offset[i]), (block_col * blocksize + total_node_offset[j]))
                        new_edges.extend(random_edge)


    # print(keep_cnt)
    # print("Edge raw {} new {}".format(len(edge_index[0]), len(new_edges)))
    # print(new_edges)
    return torch.tensor(new_edges).t()


def reformat_graph(g, edge_index, k=4, block_size=16, beta_coeffi='1'):

    # logging.getLogger('dgl').setLevel(logging.WARNING)
    t0 = time.time()
    with suppress_stdout():
        partition_ids = dgl.metis_partition_assignment(g, k)
    # exit(0)
    t1 = time.time()
    new_id_mapping = np.empty(g.num_nodes(), dtype=np.int64)
    # print(x.shape)
    current_id = 0
    for part_id in range(k):
        nodes_in_part = np.where(partition_ids == part_id)[0]
        new_id_mapping[nodes_in_part] = np.arange(current_id, current_id + len(nodes_in_part))
        current_id += len(nodes_in_part)
        
    t2 = time.time()

    p_ori = len(edge_index[0]) / (g.num_nodes() * g.num_nodes()) # 1-sparsity
    if beta_coeffi == '1':
        p = 1
    else:
        p = beta_coeffi * p_ori
    p = 0
    
    new_edge_index = generate_new_edges_optimized(edge_index, k, partition_ids, p, blocksize=block_size, new_id_mapping=new_id_mapping)
    # new_edge_index = (new_id_mapping[src.numpy()], new_id_mapping[dst.numpy()])
    # new_edge_index = (new_id_mapping[block_src.numpy()], new_id_mapping[block_dst.numpy()])
    t3 = time.time()
    new_id_mapping_tensor = torch.from_numpy(new_id_mapping)

    sorted_indices = torch.argsort(new_id_mapping_tensor)

    sorted_indices_edge = torch.argsort(new_edge_index[0])

    sorted_edge_index = new_edge_index[:, sorted_indices_edge]
    t4 = time.time()
    # print("Time in reorder {} {} {}".format(t1-t0, t2-t1, t3-t2))
    return sorted_edge_index, sorted_indices


def split_mb_in_advance(num_batches, schedule, packed_batch, device, k, reorder_graph=True):
    preprocessed_batches = []
    for i in range(num_batches):
        g_i, features_i, labels_i = packed_batch[i]
        if dist.get_rank() == 0:
            inputs = (g_i, features_i)
        else:
            inputs = (g_i, )
        args_split, kwargs_split, chunked_sg_ori_node_idxes = schedule._split_inputs(inputs)
        targets_split = list(torch.tensor_split(labels_i, schedule._n_microbatches))
        reordered_targets = []

        if reorder_graph:
            for i, args_chunk in enumerate(args_split):
                chunk_g = args_chunk[0]
                src, dst = chunk_g.edges()  # 分别获取起点和终点的 tensor
                edge_index = torch.stack([src, dst], dim=0)  
                sorted_edge_index, sorted_indices = reformat_graph(chunk_g, edge_index, k)
                new_src = sorted_edge_index[0]
                new_dst = sorted_edge_index[1]
                new_chunk_g = dgl.graph((new_src, new_dst), num_nodes=chunk_g.num_nodes()).to(device)
                processed_args = [new_chunk_g]
                if len(args_chunk) > 1:
                    for arg in args_chunk[1:]:
                        if isinstance(arg, torch.Tensor):
                            processed_args.append(arg[sorted_indices].to(device))
                        else:
                            processed_args.append(arg)
                
                args_split[i] = tuple(processed_args)
                chunked_sg_ori_node_idxes[i] = chunked_sg_ori_node_idxes[i][sorted_indices].to(device)
                reordered_targets.append(targets_split[i][sorted_indices].to(device))
        else:
            reordered_targets = targets_split

        # Save the processed batch
        preprocessed_batches.append({
        'args_split': args_split,
        'kwargs_split': kwargs_split,
        'chunked_sg_ori_node_idxes': chunked_sg_ori_node_idxes,
        'reordered_targets': reordered_targets,
        })
    return preprocessed_batches



if __name__ == "__main__":
    g, split_idx, data_x, data_y, num_classes = get_dataset('pubmed')