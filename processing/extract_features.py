import os
import json
import argparse
import openslide
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging

import torch
import torchvision.transforms as torch_trans
from torchvision import models
from torchvision.models import (
    VGG16_Weights,
    ResNet18_Weights,
    ResNet50_Weights
)
import torch.nn as nn
from .utils import save_hdf5 # Assuming utils.py with save_hdf5 is in the same directory

# Logger setup will be called in main()
logger = logging.getLogger(__name__)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def create_encoder(args):
    logger.info(f"Creating encoder: {args.image_encoder}")
    if args.image_encoder == 'vgg16':
        encoder = models.vgg16(weights=VGG16_Weights.DEFAULT).to(args.device)
        # Remove the last few layers of the classifier to get features
        encoder.classifier = nn.Sequential(*list(encoder.classifier.children())[:-3])

    elif args.image_encoder == 'resnet50':
        encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(args.device)
        # Remove the final fully connected layer and add Flatten
        layers = list(encoder.children())[:-1] 
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)

    elif args.image_encoder == 'resnet18':
        # Create the target ResNet18 feature extractor structure first.
        # This structure will have layers indexed 0 through 8 (conv1 to avgpool)
        # followed by Flatten at index 9.
        
        # Initialize with default ImageNet weights if no custom weights are provided,
        # otherwise initialize with no weights as custom ones will be loaded.
        if args.encoder_weights is None:
            logger.info("Using default ImageNet ResNet18 weights.")
            temp_resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            logger.info("Initializing ResNet18 structure to load custom weights.")
            temp_resnet = models.resnet18(weights=None) # No weights, just structure

        resnet_layers = list(temp_resnet.children())[:-1] # All layers except the original FC
        feature_extractor_layers = resnet_layers + [nn.Flatten(1)]
        encoder = nn.Sequential(*feature_extractor_layers).to(args.device)

        if args.encoder_weights is not None:
            logger.info(f"Loading custom SimCLR encoder weights from {args.encoder_weights}")
            state_dict = torch.load(args.encoder_weights, map_location=args.device)
            
            # Handle common checkpoint structures (e.g., if 'state_dict' or 'model' is a top-level key)
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict: 
                    state_dict = state_dict['model']

            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                
                # 1. Strip 'module.' prefix (from DataParallel/DDP)
                if new_k.startswith('module.'):
                    new_k = new_k[len('module.'):]
                
                # 2. Check if it's a SimCLR projection head key (e.g., 'l1.weight', 'l2.bias')
                # These should be skipped as we only want the backbone.
                if new_k.startswith('l1.') or new_k.startswith('l2.'): 
                    logger.debug(f"Skipping SimCLR projection head layer: {k} (as {new_k})")
                    continue
                    
                # 3. Strip 'features.' prefix (specific to your SimCLR checkpoint structure)
                if new_k.startswith('features.'):
                    new_k = new_k[len('features.'):]
                else:
                    # If after stripping 'module.', it doesn't start with 'features.' or known projection head keys,
                    # it might be an unexpected key for this specific SimCLR checkpoint structure.
                    # However, some ResNet layers might not be under 'features.' if the saving was different.
                    # For now, we assume backbone weights are under 'features.' after 'module.'
                    # If a key is neither projection head nor 'features.', log a warning but still try to include it
                    # if it matches the target encoder structure.
                    # A more robust way would be to map known ResNet layer names.
                    # Given your key dump, 'features.' is the main prefix for backbone layers.
                    # If a key is NOT 'l1/l2' and NOT 'features.X', it's likely not part of the ResNet backbone
                    # as per your SimCLR checkpoint structure.
                    logger.warning(f"Key '{k}' (processed to '{new_k}') does not start with 'features.' after 'module.' stripping, and is not l1/l2. Skipping this key for SimCLR loading.")
                    continue
                
                cleaned_state_dict[new_k] = v
            
            # The `encoder` is an nn.Sequential. Its layers are numerically indexed:
            # encoder[0] is conv1
            # encoder[1] is bn1
            # ...
            # encoder[4] is layer1
            # ...
            # encoder[8] is avgpool 
            # The keys in cleaned_state_dict should now be like '0.weight', '4.0.conv1.weight', etc.,
            # which directly map to the numerically indexed layers of the `encoder`.
            
            missing_keys, unexpected_keys = encoder.load_state_dict(cleaned_state_dict, strict=False)
            
            logger.info(f"Custom weights loaded into ResNet18. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            if missing_keys:
                logger.debug(f"Missing keys during load: {missing_keys}") 
            if unexpected_keys:
                logger.warning(f"Unexpected keys during load: {unexpected_keys}") 
            
            # Verification: Check if critical weights were loaded.
            # If '0.weight' (conv1) is missing from cleaned_state_dict, it's a major issue.
            if not any(k.startswith('0.') for k in cleaned_state_dict.keys()) and len(missing_keys) > (len(encoder) - 5): # Heuristic
                 logger.error("It seems critical weights (e.g., for conv1, mapped to '0.weight') were not found or loaded. "
                              "Please verify the SimCLR checkpoint key structure and the stripping logic.")
    else:
        raise ValueError(f"Invalid image_encoder: {args.image_encoder}")
    
    logger.debug(f"{args.image_encoder} architecture:\n{encoder}")
    encoder.eval() # Set encoder to evaluation mode
    return encoder

def extract_patch_features(args, image_rgb, encoder_model, transform=None): # Renamed to avoid conflict
    """Extracts features from a single patch image."""
    with torch.no_grad():
        if transform is None:
            # Default transform: ToTensor and then Normalize for ImageNet
            # For SimCLR, normalization might have been different during its pretraining.
            # If SimCLR used specific normalization, it should be applied here.
            # For now, using standard ImageNet normalization as a general case.
            transform_list = [torch_trans.ToTensor()]
            if not args.encoder_weights: # Apply ImageNet normalization if using default weights
                 transform_list.append(torch_trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            transform_to_apply = torch_trans.Compose(transform_list)
            img_tensor = transform_to_apply(image_rgb).unsqueeze(dim=0).to(args.device)
        else:
            # If a custom transform is provided (e.g. via args in future)
            img_tensor = transform(image_rgb).unsqueeze(dim=0).to(args.device)

        feat = encoder_model(img_tensor).cpu().numpy()
        return feat

def extract_slide_features(args, current_encoder, feat_dir_path): # Renamed to avoid conflict
    """Extracts features for all slides based on their coordinate files."""
    coord_dir = Path(args.patch_dir) / 'coord'
    if not coord_dir.exists():
        logger.error(f"Coord directory does not exist: {coord_dir}")
        return

    h5_dir = Path(feat_dir_path) / 'h5_files'
    pt_dir = Path(feat_dir_path) / 'pt_files'
    npz_dir = Path(feat_dir_path) / 'npz_files'
    h5_dir.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    coord_file_list = sorted(list(coord_dir.glob('*.json')))
    logger.info(f"Found {len(coord_file_list)} coord files in {coord_dir}")

    for i, coord_filepath in enumerate(coord_file_list):
        filename_stem = coord_filepath.stem
        h5_filepath = h5_dir / f'{filename_stem}.h5'
        pt_filepath = pt_dir / f'{filename_stem}.pt'
        npz_filepath = npz_dir / f'{filename_stem}.npz'

        if npz_filepath.exists() and not args.exist_ok:
            logger.info(f"{npz_filepath.name} exists, skipping...")
            continue

        try:
            with open(coord_filepath) as fp:
                coord_dict = json.load(fp)
        except Exception as e:
            logger.error(f"Error loading JSON from {coord_filepath}: {e}")
            continue

        num_patches = coord_dict.get('num_patches', 0)
        if num_patches == 0:
            logger.warning(f"{filename_stem} has 0 patches or num_patches key missing, skipping...")
            continue
        
        slide_filepath_str = coord_dict.get('slide_filepath')
        if not slide_filepath_str or not Path(slide_filepath_str).exists():
            logger.error(f"Slide filepath missing or invalid in {coord_filepath.name}: {slide_filepath_str}. Skipping slide.")
            continue

        num_row = coord_dict.get('num_row')
        num_col = coord_dict.get('num_col')
        coords = coord_dict.get('coords', [])
        patch_size_level0 = coord_dict.get('patch_size_level0')
        patch_size_target = coord_dict.get('patch_size') # The size patches were resized to if different from level0

        if None in [num_row, num_col, patch_size_level0, patch_size_target] or not coords:
            logger.error(f"Essential coordinate metadata missing in {coord_filepath.name}. Skipping slide.")
            continue

        try:
            slide = openslide.open_slide(slide_filepath_str)
        except Exception as e:
            logger.error(f"Error opening slide {slide_filepath_str} for {filename_stem}: {e}")
            continue
            
        coords_pbar = tqdm(coords, desc=f"{i + 1:3}/{len(coord_file_list):3} | {filename_stem}")
        features_list, actual_coords_list = [], []

        for patch_coord_info in coords_pbar:
            try:
                img_patch_rgb = slide.read_region(
                    location=(patch_coord_info['x'], patch_coord_info['y']),
                    level=0, # Assuming coordinates are for level 0
                    size=(patch_size_level0, patch_size_level0)
                ).convert('RGB')
                
                # Resize if the target patch_size for feature extraction is different
                if (patch_size_level0, patch_size_level0) != (patch_size_target, patch_size_target):
                    img_patch_rgb = img_patch_rgb.resize((patch_size_target, patch_size_target))

                feat_vector = extract_patch_features(args, img_patch_rgb, current_encoder)
                features_list.append(feat_vector)
                actual_coords_list.append(np.array([patch_coord_info['row'], patch_coord_info['col']], dtype=int))
            except Exception as e:
                logger.error(f"Error processing patch at coord {patch_coord_info} for slide {filename_stem}: {e}")
                continue # Skip this patch

        if not features_list:
            logger.warning(f"No features extracted for slide {filename_stem}. Skipping save.")
            slide.close()
            continue

        img_features_np = np.concatenate(features_list, axis=0)
        actual_coords_np = np.stack(actual_coords_list, axis=0)

        # Save .h5 file
        asset_dict = {'features': img_features_np, 'coords': actual_coords_np}
        try:
            save_hdf5(str(h5_filepath), asset_dict, attr_dict=None, mode='w')
        except Exception as e:
            logger.error(f"Error saving HDF5 {h5_filepath.name}: {e}")

        # Save .pt file
        try:
            torch.save(torch.from_numpy(img_features_np), str(pt_filepath))
        except Exception as e:
            logger.error(f"Error saving PT {pt_filepath.name}: {e}")
            
        # Save .npz file
        try:
            np.savez(
                file=str(npz_filepath),
                filename=filename_stem,
                num_patches=len(features_list), # Use actual number of extracted features
                num_row=num_row,
                num_col=num_col,
                img_features=img_features_np,
                coords=actual_coords_np
            )
            logger.info(f"Saved: {h5_filepath.name}, {pt_filepath.name}, {npz_filepath.name} ({len(features_list)} patches)")
        except Exception as e:
            logger.error(f"Error saving NPZ {npz_filepath.name}: {e}")
        
        slide.close()


def run_extraction(args): # Renamed from run to avoid conflict if this script is imported
    if args.device != 'cpu':
        # Ensure CUDA_VISIBLE_DEVICES is set if multiple GPUs are available
        # and a specific one is chosen.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        current_device = torch.device('cpu')
    args.device = current_device # Update args.device to be the torch.device object
    logger.info(f"Using device: {args.device}")


    if args.encoder_weights is not None:
        # Use a more descriptive name based on the weights file stem
        encoder_folder_name = Path(args.encoder_weights).stem
    else:
        encoder_folder_name = args.image_encoder

    if args.feat_dir is not None:
        # User specified a top-level feature directory
        final_feat_dir = Path(args.feat_dir) / encoder_folder_name
    else:
        # Default: save under patch_dir/features/encoder_name
        final_feat_dir = Path(args.patch_dir) / 'features' / encoder_folder_name

    final_feat_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving features to {final_feat_dir}")

    encoder_model = create_encoder(args)
    extract_slide_features(args, encoder_model, feat_dir_path=final_feat_dir)

def main():
    setup_logger() # Call logger setup
    parser = argparse.ArgumentParser(description="Extract features from WSI patches.")
    parser.add_argument('--patch_dir', type=str, required=True, 
                        help='Directory containing `coord` subdirectory with JSON coordinate files.')
    parser.add_argument('--feat_dir', type=str, default=None, 
                        help='Top-level directory to save features. A subfolder based on encoder name will be created. '
                             'Defaults to <patch_dir>/features/<encoder_name>.')
    parser.add_argument('--image_encoder', type=str, default='resnet18',
                        choices=['vgg16', 'resnet18', 'resnet50'], 
                        help='CNN model to use as feature extractor.')
    parser.add_argument('--encoder_weights', type=str, default=None, 
                        help='Path to custom encoder weights (.pth file). If None, uses default ImageNet weights.')
    parser.add_argument('--device', type=str, default='0', 
                        help='CUDA device index (e.g., "0", "1") or "cpu".')
    parser.add_argument('--exist_ok', action='store_true', default=False,
                        help='If set, overwrite existing feature files. Default is to skip.')
    
    args = parser.parse_args()

    # Basic validation
    if not Path(args.patch_dir).exists() or not (Path(args.patch_dir) / 'coord').exists():
        logger.error(f"Patch directory or its 'coord' subdirectory not found: {args.patch_dir}")
        return

    torch.set_num_threads(min(8, os.cpu_count() or 1)) # Set PyTorch threads for CPU operations
    run_extraction(args)
    logger.info("Feature extraction process completed.")

if __name__ == '__main__':
    main()