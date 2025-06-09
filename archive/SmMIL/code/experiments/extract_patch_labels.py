import argparse
import os

import pandas as pd
import numpy as np

import h5py
import zarr
import tifffile

from tqdm import tqdm


def extract_labels(mask, coords, patch_size, wsi_label):

    if wsi_label==0:
        labels_array = np.zeros(len(coords))
        print(f'{wsi_name} is normal')
    else:
        threshold = 0.5
        repeat = True
        while repeat:
            labels = []
            pbar = tqdm(total=len(coords))
            pbar.set_description(f'{wsi_name}')
            for coord in coords:
                x, y = coord

                # 0: background, 1: tissue (normal), 2: tumor
                # Remember that the mask is transposed
                patch = mask[y:y+patch_size, x:x+patch_size]
                patch_max = np.max(patch)
                patch_min = np.min(patch)
                if patch_max > 2 or patch_min < 0:
                    raise ValueError('Error in the patch with coordinates: ', coord)
                
                patch = np.where(patch == 2, 1, 0)
                
                num_c = np.sum(patch)
                num_pixels = patch_size*patch_size

                if num_c/num_pixels > threshold:
                    label = 1
                else:
                    label = 0
                
                labels.append(label)
                pbar.update(1)
            pbar.close()
            labels_array = np.array(labels)
            num_tumor = np.sum(labels_array)
            repeat = num_tumor == 0
            if repeat:
                print(f'{wsi_name} is tumor but has no tumor patches. Repeating the process with a new threshold.')
                threshold -= 0.1
    
    return labels_array

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', default='/data/wsi_labels.csv', type=str, help="CSV with WSI labels")
parser.add_argument('--masks_dir', default='/data/masks/', type=str, help="Directory with WSI masks in .tif format")
parser.add_argument('--coords_dir', default='/data/coords/', type=str, help="Directory with WSI coordinates")
parser.add_argument('--save_dir', default='/data/patch_labels/', type=str, help="Save dir")

args = parser.parse_args()

print('Arguments:')
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

wsi_names = os.listdir(args.coords_dir)
wsi_names = [ wsi_name.split('.')[0] for wsi_name in wsi_names ]
wsi_names = sorted(wsi_names)

df_wsi_labels = pd.read_csv(args.csv_path)
df_wsi_labels['wsi_name'] = df_wsi_labels['wsi_name'].apply(lambda x: x.split('.')[0])
wsi_labels = [ int(df_wsi_labels[df_wsi_labels['wsi_name'] == wsi_name]['wsi_label'].values[0]) for wsi_name in wsi_names ]

for wsi_name, wsi_label in zip(wsi_names, wsi_labels):

    save_labels_path = os.path.join(args.save_dir, wsi_name)

    if os.path.exists(save_labels_path + '.npy'):
        print(f'{wsi_name} already exists. Skipping...')
        continue
    
    wsi_path = os.path.join(args.coords_dir, wsi_name + '.h5')
    f = h5py.File(wsi_path, 'r+')
    wsi_file = f['coords']

    patch_size = wsi_file.attrs['patch_size']
    patch_level = wsi_file.attrs['patch_level']
    downsample = wsi_file.attrs['downsample']
    coords = np.array(wsi_file) # (n_patches, 2)
    coords = coords // downsample
    coords = coords.astype(int)

    mask_path = os.path.join(args.masks_dir, wsi_name + '_mask.tif')
    store = tifffile.imread(mask_path, aszarr=True)
    z = zarr.open(store, mode='r')
    mask = z[patch_level]

    labels = extract_labels(mask, coords, patch_size, wsi_label)

    # Save the labels
    
    np.save(save_labels_path, labels)


