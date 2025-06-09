from sklearn.model_selection import train_test_split

from datasets import RSNADataset, PandaDataset, CamelyonDataset

CLUSTER_DATA_DIR = '/data/datasets'
HOME_DATA_DIR = '/home/fran/data/datasets'

def load_dataset(config, mode='train_val'):
    name = config.dataset_name
    val_prop = config.val_prop
    n_samples = None
    seed = config.seed
    if "rsna" in name:

        # rsna-features_<model_name>
        features_dir_name = name.split('-')[1]
        data_path = f'{HOME_DATA_DIR}/RSNA_ICH/features/{features_dir_name}/'
        processed_data_path = f'{HOME_DATA_DIR}/RSNA_ICH/MIL_processed/{features_dir_name}/'

        if mode == 'train_val':
            csv_path = f'{CLUSTER_DATA_DIR}/RSNA_ICH/bags_train.csv'
        else:
            csv_path = f'{CLUSTER_DATA_DIR}/RSNA_ICH/bags_test.csv'

        dataset = RSNADataset(
            data_path=data_path, 
            processed_data_path=processed_data_path, 
            csv_path=csv_path, 
            n_samples=n_samples, 
            use_slice_distances=config.use_inst_distances, 
        )

    elif "panda" in name:

        # panda-<patch_dir>-<features_dir>
        # Ex: panda-patches_512-features_resnet18

        patch_dir = name.split('-')[1]

        features_dir_name = name.split('-')[2]
        data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/features/{features_dir_name}/'
        processed_data_path = f'{HOME_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/processed/{features_dir_name}/'
        aux_csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/wsi_labels.csv'

        if mode == 'train_val':
            csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/train_val_patches.csv'
        else:
            csv_path = f'{CLUSTER_DATA_DIR}/PANDA/PANDA_downsample/{patch_dir}/test_patches.csv'
    
        dataset = PandaDataset(
            data_path=data_path, 
            processed_data_path=processed_data_path, 
            csv_path=csv_path, 
            aux_csv_path=aux_csv_path,
            n_samples=n_samples, 
            use_patch_distances=config.use_inst_distances, 
        )
    
    elif 'camelyon16' in name:

        # camelyon16-<patch_dir>-<features_dir>
        # Ex: camelyon16-patches_512_preset-features_resnet50_bt

        patches_dir_name = name.split('-')[1]
        features_dir_name = name.split('-')[2]
        main_data_path = f'{HOME_DATA_DIR}/CAMELYON16/{patches_dir_name}/'

        if mode == 'train_val':
            csv_path = f'{CLUSTER_DATA_DIR}/CAMELYON16/original/train.csv'
        else:
            csv_path = f'{CLUSTER_DATA_DIR}/CAMELYON16/original/test.csv'

        dataset = CamelyonDataset(
            main_data_path, 
            csv_path, 
            features_dir_name, 
            use_patch_distances=config.use_inst_distances, 
        )

    else:
        raise ValueError(f"Dataset {name} not supported")

    if mode == 'train_val':

        bags_labels = dataset.get_bag_labels()
        len_ds = len(bags_labels)
        
        idx = list(range(len_ds))
        idx_train, idx_val = train_test_split(idx, test_size=val_prop, random_state=seed, stratify=bags_labels)

        train_dataset = dataset.subset(idx_train)
        val_dataset = dataset.subset(idx_val)

        return train_dataset, val_dataset
    
    else:
        return dataset
