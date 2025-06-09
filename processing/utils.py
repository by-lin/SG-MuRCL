import numpy as np
import json
import h5py
from pathlib import Path
import logging


def setup_logger():
    """Sets up a basic console logger for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def get_three_points(x_step, y_step, size):
    top_left = (int(x_step * size), int(y_step * size))
    bottom_right = (int(top_left[0] + size), int(top_left[1] + size))
    center = (int((top_left[0] + bottom_right[0]) // 2), int((top_left[1] + bottom_right[1]) // 2))
    return top_left, bottom_right, center


def downsample_image(slide, downsampling_factor=16, mode="numpy"):
    best_downsampling_level = slide.get_best_level_for_downsample(downsampling_factor + 0.1)

    # Get the image at the requested scale
    svs_native_levelimg = slide.read_region((0, 0), best_downsampling_level,
                                            slide.level_dimensions[best_downsampling_level])
    target_size = tuple([int(x // downsampling_factor) for x in slide.dimensions])
    img = svs_native_levelimg.resize(target_size)

    # By default, return a numpy array as RGB, otherwise, return PIL image
    if mode == "numpy":
        # Remove the alpha channel
        img = np.array(img.convert("RGB"))

    return img, best_downsampling_level


def keep_patch(mask_patch, thres, bg_color):
    """Specialized selector for otsu or adaptive TileGenerator.

    Determines if a mask tile contains a certain percentage of foreground.

    Args:
        mask_patch: Numpy array for the current mask tile.
        thres: Float indicating the minimum foreground content [0, 1] in
            the patch to select the tile.
        bg_color: Numpy array with the background color for the mask.

    Returns:
        _: Integer [0/1] indicating if the tile has been selected or not.
    """
    # print(mask_patch.shape)
    # print(bg_color.shape)
    bg = np.all(mask_patch == bg_color, axis=2)
    # print(bg.shape)
    bg_proportion = np.sum(bg) / bg.size

    if bg_proportion <= (1 - thres):
        output = 1
    else:
        output = 0

    return output


def out_of_bound(w, h, x, y):
    return x >= w or y >= h


def dump_json(data_dict, filename):
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(data_dict, fp)


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        data_dict = json.load(fp)
    return data_dict


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path

def auto_detect_clustering_paths(data_csv_path, graph_level='patch'):
    """
    Simple auto-detection: find clustering folders in same directory as CSV.
    
    Args:
        data_csv_path (str): Path to CSV file
        graph_level (str): 'patch' or 'region'  
        
    Returns:
        tuple: (data_root_dir, clustering_folder_name)
    """
    csv_path = Path(data_csv_path)
    
    # Data root is same directory as CSV
    data_root = csv_path.parent
    
    # Clustering folder name based on level
    if graph_level == 'patch':
        clustering_folder = f'k-means-10'
    else:  # region
        clustering_folder = f'k-regions-10'
    
    logger.info(f"Auto-detected:")
    logger.info(f"  Data root: {data_root}")
    logger.info(f"  Clustering folder: {clustering_folder}")
    
    return str(data_root), clustering_folder