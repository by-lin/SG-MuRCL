import torch
import os
import numpy as np
import time

from tqdm import tqdm

from .MILDataset import MILDataset

class ProcessedMILDataset(MILDataset):
    """
    A MIL dataset that loads pre-processed data from disk. 
    - If the data is not found, it will be processed and saved. In this case, the data_path is needed.
    - If the adjacency matrix data is not found, it will be built and saved.
    """
    def __init__(self, 
                 processed_data_path : str,
                 data_path : str = None,
                 keep_in_memory : bool = False,
                 **kwargs,
                 ):
        
        if not hasattr(self, 'dataset_name'):
            self.dataset_name = 'ProcessedMILDataset'
        
        self.processed_data_path = processed_data_path
        self.data_path = data_path
        self.keep_in_memory = keep_in_memory
        self.processed = False

        super(ProcessedMILDataset, self).__init__(**kwargs)

        self.processed = self._check_already_processed()

        if not self.processed:
            if self.data_path is None:
                raise ValueError(f'[{self.dataset_name}] data_path needed to process data!')
            self._process_data()
            self.processed = True
        
        if self.data_shape is None:
            self.data_shape = self._compute_data_shape()

        # Loaded dict: { bag_name: True/False }
    
    def _bag_loader(self, path):
        if path.endswith('.npy'):
            d = np.load(path)
        else:
            raise ValueError(f'[{self.dataset_name}] Invalid file format: {path}')
        return d

    def _compute_data_shape(self):
        if not self.processed:
            return None
        else:
            tmp = self._bag_loader(os.path.join(self.processed_data_path, self.bag_names[0] + '.npy'))
            return tmp.shape[1:]

    def _check_already_processed(self):
        if not os.path.exists(self.processed_data_path):
            return False
        existing_bags = os.listdir(self.processed_data_path)
        existing_bags = [ bag.split('.')[0] for bag in existing_bags ]
        existing_bags = set(existing_bags)
        existing_bags = existing_bags.intersection(set(self.bag_names))
        print(f'[{self.dataset_name}] Found {len(existing_bags)} already processed bags')
        return len(existing_bags) == len(self.bag_names)
            
    def _process_data(self):
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        pbar = tqdm(self.bag_names, total=len(self.bag_names))
        pbar.set_description(f'[{self.dataset_name}] Processing and saving data')
        for bag_name in pbar:
            bag_feat = self._load_bag_feat(bag_name)
            np.save(os.path.join(self.processed_data_path, bag_name + '.npy'), bag_feat)

    def _load_bag_feat(self, bag_name):

        if not self.processed:
            # If not processed, load instance by instance
            return super(ProcessedMILDataset, self)._load_bag_feat(bag_name)
        else:   
            bag_feat = self._bag_loader(os.path.join(self.processed_data_path, bag_name + '.npy'))

            return bag_feat
    
    def _get_bag_feat(self, bag_name):
        if 'bag_feat' in self.data_dict[bag_name]:
            bag_feat = self.data_dict[bag_name]['bag_feat']
        else:
            bag_feat = self._load_bag_feat(bag_name)
            len_bag_feat = len(bag_feat)
            if len_bag_feat != len(self.data_dict[bag_name]['inst_labels']):
                raise ValueError(f'[{self.dataset_name}] Bag size mismatch: {len_bag_feat} != {self.bag_sizes[bag_name]}')
            if self.keep_in_memory:
                self.data_dict[bag_name]['bag_feat'] = bag_feat
        return bag_feat