from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
from torch_geometric.utils import degree


def get_raw_text_arxiv(use_text=False, seed=0, group_by_degree=True):
    data_dir = '/home/mzhang/work/TAPE/dataset/'
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', transform=T.ToSparseTensor(), root = data_dir)
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data.edge_index = data.adj_t.to_symmetric()

    # Group nodes by degree
    if group_by_degree:
        # Calculate node degrees using the adjacency tensor
        row, col, _ = data.adj_t.coo()
        node_degrees = torch.bincount(row, minlength=data.num_nodes)
        
        # Sort nodes by degree in descending order
        sorted_indices = torch.argsort(node_degrees, descending=True)
        
        # Split into three groups of approximately equal size
        num_groups = 3
        group_size = data.num_nodes // num_groups
        remaining = data.num_nodes % num_groups
        
        # Adjust group sizes to handle remainder
        group_sizes = [group_size + (1 if i < remaining else 0) for i in range(num_groups)]
        
        # Create the groups
        start_idx = 0
        degree_groups = []
        for size in group_sizes:
            end_idx = start_idx + size
            group = sorted_indices[start_idx:end_idx]
            degree_groups.append(group)
            start_idx = end_idx
            
        # Add degree groups to data object
        data.degree_groups = degree_groups

    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        data_dir+'ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv(data_dir+'ogbn_arxiv_orig/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])
    raw_text['paper id'] = pd.to_numeric(raw_text['paper id'], errors='coerce')
    raw_text = raw_text.dropna(subset=['paper id'])
    raw_text['paper id'] = raw_text['paper id'].astype('int64')
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text
