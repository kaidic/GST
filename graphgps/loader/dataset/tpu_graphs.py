from typing import Optional, Callable, List
import copy
import re
import os
import glob
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip)
from torch_geometric.utils import remove_isolated_nodes
from torch_sparse import SparseTensor

def flatten_dict(in_dict, current_key_chain=None, out_dict=None):
  """Convert dict with keys "key1.key2" to multi-level "key1": {"key2" .. }."""
  if current_key_chain is None:
    current_key_chain = []
  if out_dict is None:
    out_dict = {}
  for key, value in in_dict.items():
    if isinstance(key, tuple):
      key = '|'.join(key)

    if isinstance(value, dict):
      flatten_dict(value, current_key_chain + [key], out_dict)
    else:
      key = '.'.join(current_key_chain + [key])
      out_dict[key] = value
  return out_dict

class TPUGraphs(InMemoryDataset):

    def __init__(self, root: str, thres: int = 1000,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 source: str = 'nlp',  # 'nlp' or 'xla'
                 search: str = 'random'  # 'random' or 'default'
                ):
        assert source in ('nlp', 'xla')
        assert search in ('random', 'default')
        self.thres = thres
        self.source = source
        self.search = search
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        op_feats_mean = torch.mean(self.data.op_feats, dim=0, keepdim=True)
        op_feats_std = torch.std(self.data.op_feats, dim=0, keepdim=True)
        op_feats_std[op_feats_std < 1e-6] = 1
        self.data.op_feats = (self.data.op_feats - op_feats_mean) / op_feats_std
        
    @property
    def raw_file_names(self) -> List[str]:
        return [f'npz/layout/{self.search}/{self.source}']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_segment_{}.pt'.format(self.thres), 'split_dict_segment_{}.pt'.format(self.thres)]


    def process(self):
        data_list = []
        split_names = ['train', 'valid', 'test']
        split_dict = {'train': [], 'valid': [], 'test': []}
        graphs_cnt = 0
        parts_cnt = 0
        for raw_path in self.raw_paths:
            for split_name in split_names:
                filenames = glob.glob(osp.join(os.path.join(raw_path, split_name), '*.npz'))
                for filename in filenames:
                    split_dict[split_name].append(graphs_cnt)
                    np_file = np.load(filename)
                    np_file = flatten_dict(np_file)
                    edge_index = torch.tensor(np_file['edge_index_dict.op|feeds|op'])
                    runtime = torch.tensor(np_file['node_runtime.configs'])
                    op = torch.tensor(np_file["node_feat_dict.op"])
                    op_code = torch.tensor(np_file["node_opcode.op"])
                    config_feats = torch.tensor(np_file["node_feat_dict.configs"])
                    config_feats = config_feats.view(-1, config_feats.shape[-1])
                    config_idx = torch.tensor(np_file["node_feat_dict.config_idx"])
                    num_config = torch.tensor(np_file["num_nodes_dict.configs"])
                    num_config_idx = torch.tensor(np_file["num_nodes_dict.config_idx"])
                    num_nodes = torch.tensor(np_file["num_nodes_dict.op"])
                    num_parts = num_nodes // self.thres + 1
                    interval = num_nodes // num_parts
                    partptr = torch.arange(0, num_nodes, interval+1)
                    if partptr[-1] != num_nodes:
                        partptr = torch.cat([partptr, torch.tensor([num_nodes])])
                    data = Data(edge_index=edge_index, op_feats=op, op_code=op_code, config_feats=config_feats, config_idx=config_idx,
                                num_config=num_config, num_config_idx=num_config_idx, y=runtime, num_nodes=num_nodes, partptr=partptr, partition_idx = parts_cnt)
                    data_list.append(data)
                    graphs_cnt += 1
                    parts_cnt += num_parts * num_config
            torch.save(self.collate(data_list), self.processed_paths[0])
            torch.save(split_dict, self.processed_paths[1])
    def get_idx_split(self):
        return torch.load(self.processed_paths[1])

if __name__ == '__main__':
    dataset = TPUGraphs(root='datasets/TPUGraphs')
    import pdb; pdb.set_trace()
