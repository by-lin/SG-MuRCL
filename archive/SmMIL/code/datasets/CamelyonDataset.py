import os
import cv2
import numpy as np
import pandas as pd
import h5py


from tqdm import tqdm
from scipy.spatial import KDTree

from .ProcessedMILDataset import ProcessedMILDataset

class CamelyonDataset(ProcessedMILDataset):
    def __init__(self,
                 main_data_path,
                 csv_path,
                 features_dir_name,
                 use_patch_distances = True,
                 n_samples = None,
                 bag_size_limit = None,
                 load_at_init = False,
                 **kwargs
    ):  
        
        self.main_data_path = main_data_path
        self.features_dir_name = features_dir_name
        self.bag_size_limit = bag_size_limit
        self.load_at_init = load_at_init
        processed_data_path = main_data_path + features_dir_name + '/'
        inst_labels_path = main_data_path + 'patch_labels/'
        coords_path = main_data_path + 'coords/'     


        self.dataset_name = 'CamelyonDataset'
        super(CamelyonDataset, self).__init__(
            processed_data_path, 
            csv_path=csv_path, 
            inst_labels_path=inst_labels_path, 
            coords_path=coords_path, 
            use_patch_distances=use_patch_distances,
            n_samples=n_samples, 
            **kwargs
        )
                
        if self.n_samples is not None:
            print(f'[{self.dataset_name}] Sampling {self.n_samples} bags...')
            self.bag_names = self.bag_names[:self.n_samples]
            self.data_dict = { k : self.data_dict[k] for k in self.bag_names }

    def _inst_loader(self, path):
        if path.endswith('.npy'):
            d = np.load(path)
            # d = d[:, :1024]
        else:
            d = cv2.imread(path)
            d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
            d = d.transpose(2,0,1)
        return d

    def _get_bag_feat(self, bag_name):
        bag_feat = self._load_bag_feat(bag_name)
        bag_feat = bag_feat[self.data_dict[bag_name]['selected_idx']]
        return bag_feat

    def _init_data_dict(self):
        print(f'[{self.dataset_name}] Scanning files...')
        df = pd.read_csv(self.csv_path)
        
        bag_names_list = df['wsi_name'].apply(lambda x: x.split('.')[0]).values
        bag_labels_list = df['wsi_label'].astype(int).values

        # Obtain patch extraction atributes from first bag
        bag_name = bag_names_list[0]
        f = h5py.File(os.path.join(self.coords_path, bag_name + '.h5'), 'r')['coords']
        self.patch_size = int(f.attrs['patch_size'])
        self.patch_level = int(f.attrs['patch_level'])
        self.downsample = f.attrs['downsample']

        data_dict = {}
        num_wsis = len(bag_names_list)
        pbar = tqdm(range(num_wsis), total=num_wsis)
        for i in pbar:
            pbar.set_description(f'[{self.dataset_name}] Building data dict')

            # print(bag_labels_list[i], np.load(os.path.join(self.inst_labels_path, bag_name + '.npy')).sum())

            bag_name = bag_names_list[i]
            inst_labels = np.load(os.path.join(self.inst_labels_path, bag_name + '.npy')).astype(int)
            inst_coords = np.array(h5py.File(os.path.join(self.coords_path, bag_name + '.h5'), 'r')['coords'])
            inst_coords = inst_coords / self.downsample 
            inst_int_coords = (inst_coords / self.patch_size).astype(int)
            bag_label = bag_labels_list[i]

            if self.bag_size_limit is not None:
                bag_size = len(inst_labels)
                if bag_size > self.bag_size_limit:
                    selected_idx = np.random.choice(bag_size, self.bag_size_limit, replace=False)
                    inst_labels = inst_labels[selected_idx]
                    inst_coords = inst_coords[selected_idx]
                    inst_int_coords = inst_int_coords[selected_idx]
                else:
                    selected_idx = np.arange(bag_size)
            else:
                selected_idx = np.arange(len(inst_labels))

            data_dict[bag_name] = {
                'inst_labels': np.array(inst_labels),
                'bag_label': bag_label,
                'inst_coords': inst_coords,
                'inst_int_coords' : inst_int_coords,
                'selected_idx' : selected_idx
            }
            if self.load_at_init:
                bag_feat = self._load_bag_feat(bag_name)

                # data_dict[bag_name]['bag_feat'] = bag_feat
                edge_index, edge_weight = self._build_edge_index(inst_coords, bag_feat).astype(np.longlong)
                norm_edge_weight = self._normalize_adj_matrix(edge_index, bag_feat.shape[0])
                self.data_dict[bag_name]['edge_index'] = edge_index
                self.data_dict[bag_name]['edge_weight'] = edge_weight
                self.data_dict[bag_name]['norm_edge_weight'] = norm_edge_weight
    
        return data_dict

    def _build_edge_index(self, coords, bag_feat):
        
        kdtree = KDTree(coords)

        # Build adjacency matrix
        n_patches = len(coords)
        edge_index = []
        edge_weight = []
        for i in range(n_patches):
            
            # Self-loop
            # edge_index.append([i,i])
            # edge_weight.append(1.0)

            # Find neighboring patches within sqrt(2)*patch_size distance
            neighbors = kdtree.query_ball_point(coords[i], np.sqrt(2)*self.patch_size)
            for j in neighbors:
                if i != j:
                    edge_index.append([i,j])
                    if self.use_patch_distances:
                        dist = np.exp( -np.linalg.norm(bag_feat[i] - bag_feat[j]) / bag_feat.shape[1] )
                    else:
                        dist = 1.0
                    edge_weight.append(dist)

        edge_index = np.array(edge_index).T.astype(np.longlong) # (2, n_edges)
        edge_weight = np.array(edge_weight) # (n_edges,)
        return edge_index, edge_weight
    
    def get_class_counts(self):
        class_counts = {}
        for bag_name in self.bag_names:
            bag_label = self.data_dict[bag_name]['bag_label']
            if bag_label not in class_counts:
                class_counts[bag_label] = 1
            else:
                class_counts[bag_label] += 1
        return class_counts