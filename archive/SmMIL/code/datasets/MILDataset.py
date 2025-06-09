import torch
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence


class MILDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 **kwargs
                #  data_path : str, 
                #  csv_path : str,
                 ):
        super(MILDataset, self).__init__()

        if not hasattr(self, 'dataset_name'):
            self.dataset_name = 'MILDataset'
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # self.data_path = data_path
        # self.csv_path = csv_path

        self.data_dict = self._init_data_dict()
        self.bag_names = list(self.data_dict.keys())
        # Data dict: { bag_name: { 'bag_label': int, 'inst_paths' : [str, str, ...], 'inst_labels': array, 'adj_mat' : array} }

        self.data_shape = self._compute_data_shape()
    
    def _init_data_dict(self):
        raise NotImplementedError

    def _build_edge_index(self, *args, **kwargs):
        raise NotImplementedError
    
    def _compute_data_shape(self):
        raise NotImplementedError

    def _inst_loader(self, *args, **kwargs):
        raise NotImplementedError

    def _load_bag_feat(self, bag_name, *args, **kwargs):

        if 'inst_paths' not in self.data_dict[bag_name]:
            raise ValueError(f'[{self.dataset_name}] Instance paths not found for bag {bag_name}')

        feat_list = []
        for inst_path in self.data_dict[bag_name]['inst_paths']:
            try:
                data = self._inst_loader(inst_path)
            except Exception as e:
                print(f'[{self.dataset_name}] Error loading instance {inst_path}: {e}')
                continue
            feat_list.append(data)

        bag_feat = np.array(feat_list)
        
        return bag_feat
    
    def _get_bag_feat(self, bag_name):
        bag_feat = self._load_bag_feat(bag_name)
        return bag_feat

    def _get_edge_index(self, bag_name):
        
        bag_feat = self._get_bag_feat(bag_name)
        bag_size = bag_feat.shape[0]

        if 'edge_index' in self.data_dict[bag_name]:
            edge_index = self.data_dict[bag_name]['edge_index']
            edge_weight = self.data_dict[bag_name]['edge_weight']
            norm_edge_weight = self.data_dict[bag_name]['norm_edge_weight']
        else:
            coords = self.data_dict[bag_name]['inst_coords']
            edge_index, edge_weight = self._build_edge_index(coords, bag_feat)
            norm_edge_weight = self._normalize_adj_matrix(edge_index, edge_weight, bag_feat.shape[0])
            if bag_size == 1:
                edge_index, norm_edge_weight = self._add_self_loops(edge_index, bag_size, norm_edge_weight)
            self.data_dict[bag_name]['edge_index'] = edge_index
            self.data_dict[bag_name]['edge_weight'] = edge_weight
            self.data_dict[bag_name]['norm_edge_weight'] = norm_edge_weight
        
        return edge_index, edge_weight, norm_edge_weight
    
    def _get_adj_mat(self, bag_name):
        
        bag_size = len(self.data_dict[bag_name]['inst_labels'])

        edge_index, edge_weight, norm_edge_weight = self._get_edge_index(bag_name)
        
        adj_mat = torch.sparse_coo_tensor(edge_index, norm_edge_weight, (bag_size, bag_size)).coalesce().type(torch.float32)

        return adj_mat  
    
    def _get_max_bag_size(self):
        max_bag_size = 0
        pbar = tqdm(self.bag_names, total=len(self.bag_names))
        pbar.set_description(f'[{self.dataset_name}] Computing max bag size')
        for bag_name in pbar:
            bag_size = len(self.data_dict[bag_name]['inst_labels'])
            max_bag_size = max(max_bag_size, bag_size)
        return max_bag_size

    ########################################################################################################################################################################
    # Graph-related functions

    def _degree(self, index, edge_weight, num_nodes):
        """
        input:
            index: tensor (num_edges,)
            edge_weight: tensor (num_edges,)
        output:
            deg: tensor (num_nodes)
        """

        out = np.zeros((num_nodes))
        np.add.at(out, index, edge_weight)
        return out    
    
    def _add_self_loops(self, edge_index, num_nodes, edge_weight=None):
        """
        input:
            edge_index: tensor (2, num_edges)
        output:
            new_edge_index: tensor (2, num_edges + num_nodes)
        """

        loop_index = np.arange(0, num_nodes)
        loop_index = np.tile(loop_index, (2,1))

        if edge_index.shape[0] == 0:
            new_edge_index = loop_index
            new_edge_weight = np.ones(num_nodes)
        else:
            if edge_weight is None:
                edge_weight = np.ones(edge_index.shape[1])                 
            new_edge_index = np.hstack([edge_index, loop_index])
            new_edge_weight = np.concatenate([edge_weight, np.ones(num_nodes)])
        return new_edge_index, new_edge_weight
    
    def _remove_self_loops(self, edge_index, edge_weight=None):
        """
        input:
            edge_index: tensor (2, num_edges)
        output:
            new_edge_index: tensor (2, num_edges - num_nodes)
        """

        if edge_index.shape[0] == 0:
            return edge_index
        else:
            mask = edge_index[0] != edge_index[1]
            new_edge_index = edge_index[:,mask]
            if edge_weight is None:
                return new_edge_index
            else:
                return new_edge_index, edge_weight[mask]

    def _normalize_adj_matrix(self, edge_index, edge_weight, num_nodes):
        """
        input:
            edge_index: tensor (2, num_edges)
            edge_weight: tensor (num_edges)
        output:
            new_edge_index: tensor (2, num_edges + num_nodes)
            new_edge_weight: tensor (num_edges + num_nodes)
        """

        if edge_index.shape[0] == 0:
            new_edge_weight = np.array([])
        else:
            row = edge_index[0]
            col = edge_index[1]
            deg = self._degree(col, edge_weight, num_nodes).astype(np.float32)
            with np.errstate(divide='ignore'):
                deg_inv_sqrt = np.power(deg, -0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            # new_edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            new_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return new_edge_weight

    def _build_laplacian_matrix(self, edge_index, num_nodes):
        """
        input:
            edge_index: tensor (2, num_edges)
        output:
            L_mat: tensor (num_nodes, num_nodes)
        """

        edge_weight = self._normalize_adj_matrix(edge_index, num_nodes)
        new_edge_index, new_edge_weight = self._add_self_loops(edge_index, num_nodes, -edge_weight)

        return new_edge_index, new_edge_weight

    ########################################################################################################################################################################

    def __getitem__(self, index):        
        bag_name = self.bag_names[index]

        bag_feat = self._get_bag_feat(bag_name)

        bag_label = self.data_dict[bag_name]['bag_label']
        inst_labels = self.data_dict[bag_name]['inst_labels']        
        # bag_size = bag_feat.shape[0]

        adj_mat = self._get_adj_mat(bag_name)

        return torch.from_numpy(bag_feat).type(torch.float32), torch.as_tensor(bag_label), torch.from_numpy(inst_labels), adj_mat

    def __len__(self):
        return len(self.bag_names)
    
    ########################################################################################################################################################################

    def get_bag_labels(self):
        return [ self.data_dict[bag_name]['bag_label'] for bag_name in self.bag_names ]
    
    def subset(self, idx):

        new_dataset = deepcopy(self)
        new_dataset.bag_names = [self.bag_names[i] for i in idx]
        new_dataset.data_dict = { bag_name: self.data_dict[bag_name] for bag_name in new_dataset.bag_names }

        return new_dataset

    def collate_fn(self, batch, use_sparse=True):

        if len(batch) == 1:
            bag_data, bag_label, inst_labels, adj_mat = batch[0]
            bag_data = bag_data.unsqueeze(0)
            bag_label = bag_label.unsqueeze(0)
            inst_labels = inst_labels.unsqueeze(0)
            adj_mat = adj_mat.unsqueeze(0)
            if adj_mat.is_sparse:
                if not use_sparse:
                    adj_mat = adj_mat.to_dense()
                else:
                    adj_mat = adj_mat.coalesce()
            mask = torch.ones_like(inst_labels).float()
        else:

            batch_size = len(batch)

            bag_data_list = []
            bag_label_list = []
            inst_labels_list = []
            adj_mat_indices_list = []
            adj_mat_values_list = []
            adj_mat_shape_list = []

            for bag_data, bag_label, inst_labels, adj_mat in batch:
                bag_data_list.append(bag_data)
                bag_label_list.append(bag_label)
                inst_labels_list.append(inst_labels)
                adj_mat_indices_list.append(adj_mat.indices())
                adj_mat_values_list.append(adj_mat.values())
                adj_mat_shape_list.append(adj_mat.shape)
            
            bag_data = pad_sequence(bag_data_list, batch_first=True, padding_value=0) # (batch_size, max_bag_size, feat_dim)
            bag_label = torch.stack(bag_label_list) # (batch_size, )
            inst_labels = pad_sequence(inst_labels_list, batch_first=True, padding_value=-2) # (batch_size, max_bag_size)
            
            # bag_size = bag_data.shape[1]
            adj_mat_shape_array = np.array(adj_mat_shape_list)
            adj_mat_max_shape = tuple(np.max(adj_mat_shape_array, axis=0).astype(int))

            adj_mat_list = []
            for i in range(batch_size):
                indices = adj_mat_indices_list[i]
                values = adj_mat_values_list[i]
                adj_mat_list.append(torch.sparse_coo_tensor(indices, values, adj_mat_max_shape))
            adj_mat = torch.stack(adj_mat_list).coalesce() # (batch_size, bag_size, bag_size)
            if not use_sparse:
                adj_mat = adj_mat.to_dense()
            mask = (inst_labels != -2).float() # (batch_size, max_bag_size)

        return bag_data, bag_label, inst_labels, adj_mat, mask