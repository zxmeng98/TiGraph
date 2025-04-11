from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import json
import numpy as np
import os
import time
from .utils import time_logger

FILE = '/home/mzhang/work/TAPE/dataset/ogbn_products_orig/ogbn-products.csv'


@time_logger
def _process():
    if os.path.isfile(FILE):
        return

    print("Processing raw text...")

    data = []
    files = ['/home/mzhang/work/TAPE/dataset/ogbn_products/Amazon-3M.raw/trn.json',
             '/home/mzhang/work/TAPE/dataset/ogbn_products/Amazon-3M.raw/tst.json']
    for file in files:
        with open(file) as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    df.set_index('uid', inplace=True)

    nodeidx2asin = pd.read_csv(
        '/home/mzhang/work/TAPE/dataset/ogbn_products/mapping/nodeidx2asin.csv.gz', compression='gzip')

    dataset = PygNodePropPredDataset(
        name='ogbn-products', transform=T.ToSparseTensor())
    graph = dataset[0]
    graph.n_id = np.arange(graph.num_nodes)
    graph.n_asin = nodeidx2asin.loc[graph.n_id]['asin'].values

    graph_df = df.loc[graph.n_asin]
    graph_df['nid'] = graph.n_id
    graph_df.reset_index(inplace=True)

    if not os.path.isdir('/home/mzhang/work/TAPE/dataset/ogbn_products_orig'):
        os.mkdir('/home/mzhang/work/TAPE/dataset/ogbn_products_orig')
    pd.DataFrame.to_csv(graph_df, FILE,
                        index=False, columns=['uid', 'nid', 'title', 'content'])


def get_raw_text_products(use_text=False, seed=0, group_by_degree=4):
    data = torch.load('/home/mzhang/work/TAPE/dataset/ogbn_products/ogbn-products_subset.pt')
    text = pd.read_csv('/home/mzhang/work/TAPE/dataset/ogbn_products_orig/ogbn-products_subset.csv')
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()

    # Group nodes by degree
    if group_by_degree is not None:
        # Calculate node degrees using the adjacency tensor
        row, col, _ = data.adj_t.coo()
        node_degrees = torch.bincount(row, minlength=data.num_nodes)
        
        # Sort nodes by degree in descending order
        sorted_indices = torch.argsort(node_degrees, descending=True)
        
        # Split into three groups of approximately equal size
        num_groups = group_by_degree
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

    return data, text


if __name__ == '__main__':
    data, text = get_raw_text_products(True)
    print(data)
    print(text[0])
