from typing import Optional, Callable, List
import copy
import os
import glob
import os.path as osp

import torch
from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.utils import remove_isolated_nodes
from torch_sparse import SparseTensor

class MalNetLarge(InMemoryDataset):

    def __init__(self, root: str, thres: int = 5000,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.thres = thres
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        folders = ['addisplay', 'adware', 'benign', 'downloader', 'trojan']
        return [osp.join('malnet-graphs-large', folder) for folder in folders]


    @property
    def processed_file_names(self) -> List[str]:
        return ['data_segment_{}.pt'.format(self.thres), 'split_dict_segment_{}.pt'.format(self.thres)]


    def process(self):
        data_list = []
        split_dict = {'train': [], 'valid': [], 'test': []}

        parse = lambda f: set([x.split('/')[-1]
                               for x in f.read().split('\n')[:-1]])  # -1 for empty line at EOF
        split_dir = osp.join(self.raw_dir, 'split_info_large', 'type')
        with open(osp.join(split_dir, 'train.txt'), 'r') as f:
            train_names = parse(f)
            assert len(train_names) == 3500
        with open(osp.join(split_dir, 'val.txt'), 'r') as f:
            val_names = parse(f)
            assert len(val_names) == 500
        with open(osp.join(split_dir, 'test.txt'), 'r') as f:
            test_names = parse(f)
            assert len(test_names) == 1000

        for y, raw_path in enumerate(self.raw_paths):
            raw_paths = [osp.join(raw_path, d) for d in os.listdir(raw_path)]
            filenames = []
            print('Loading {}...'.format(y))
            for raw_path in raw_paths:
                filenames.extend(glob.glob(osp.join(raw_path, '*.edgelist')))
            for filename in filenames:
                with open(filename, 'r') as f:
                    edges = f.read().split('\n')[5:-1]
                edge_index = [[int(s) for s in edge.split()] for edge in edges]
                edge_index = torch.tensor(edge_index).t().contiguous()
                edge_index = remove_isolated_nodes(edge_index)[0]
                num_nodes = int(edge_index.max()) + 1
                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)

                ind = len(data_list) - 1
                graph_id = osp.splitext(osp.basename(filename))[0]
                if graph_id in train_names:
                    split_dict['train'].append(ind)
                elif graph_id in val_names:
                    split_dict['valid'].append(ind)
                elif graph_id in test_names:
                    split_dict['test'].append(ind)
                else:
                    raise ValueError(f'No split assignment for "{graph_id}".')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Segment each graph into subgraphs if needed.
        new_data_list = []
        parts_cnt = 0
        for idx, graph in enumerate(data_list):
            
            N, E = graph.num_nodes, graph.num_edges
            adj = SparseTensor(
                row=graph.edge_index[0], col=graph.edge_index[1],
                value=torch.arange(E, device=graph.edge_index.device),
                sparse_sizes=(N, N))
            adj = adj.to_symmetric()
            num_partition = N // self.thres + 1
            adj, partptr, perm = adj.partition(num_partition, False)
            out = copy.copy(graph)
            for key, value in graph.items():
                if graph.is_node_attr(key):
                    out[key] = value[perm]

            out.edge_index = None
            out.adj = adj
            row, col, val = adj.coo()
            out.edge_index = torch.stack([row, col], dim=0)
            out.partptr = partptr
            out.idx = idx
            out.partition_idx = parts_cnt
            new_data_list.append(out)
            parts_cnt += num_partition
            print(f'Graph {idx} is segmented into {num_partition} subgraphs.')
        torch.save(self.collate(new_data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])

    def get_idx_split(self):
        return torch.load(self.processed_paths[1])
