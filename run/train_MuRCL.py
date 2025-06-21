import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import yaml
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import logging

from datasets.datasets import SGMuRCLDataset, mixup, get_selected_bag_and_graph
from utils.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, init_seeds, save_checkpoint, load_json
from models import rlmil, abmil, cl, smtabmil
from utils.losses import NT_Xent

# Add these imports for GAT and pipeline
from models.graph_encoders import BatchedGATWrapper
from models.pipeline_modules import GraphAndMILPipeline

# Logger Setup
def setup_logger():
    """Sets up a basic console logger for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def create_save_dir(args):
    """Create directory to save experiment results by global arguments."""
    dir1 = f"{args.dataset}_np_{args.feat_size}"
    dir2 = f"MuRCL"
    murcl_setting = [
        f"T{args.T}",
        f"pd{args.projection_dim}",
        f"as{args.action_std}",
        f"pg{args.ppo_gamma}",
        f"tau{args.temperature}",
        f"alpha{args.alpha}",
    ]
    
    # Add graph encoder settings to directory name
    if args.graph_encoder_type != "none":
        murcl_setting.extend([
            f"gnn{args.graph_encoder_type}",
            f"ghd{args.gnn_hidden_dim}",
            f"god{args.gnn_output_dim}",
            f"gnl{args.gnn_num_layers}",
        ])
        if args.graph_encoder_type == "gat":
            murcl_setting.append(f"gah{args.gat_heads}")
    
    dir3 = "_".join(murcl_setting)
    
    # Update arch name to include MIL aggregator type
    if args.graph_encoder_type != "none":
        dir4 = f"{args.graph_encoder_type.upper()}-{args.mil_aggregator_type.upper()}"
    else:
        dir4 = args.mil_aggregator_type.upper()
    
    # Arch Setting based on MIL aggregator type
    if args.mil_aggregator_type == "abmil":
        arch_setting = [
            f"L{args.model_dim}",
            f"D{args.D}",
            f"dpt{args.dropout}",
        ]
        if hasattr(args, "abmil_K"):
            arch_setting.append(f"K{args.abmil_K}")
    elif args.mil_aggregator_type == "smtabmil":
        arch_setting = [
            f"L{args.model_dim}",
            f"D{args.D}",
            f"dpt{args.dropout}",
            f"sma{args.sm_alpha}",
            f"smw{args.sm_where}",
            f"sms{args.sm_steps}",
            f"th{args.transf_num_heads}",
            f"tl{args.transf_num_layers}",
        ]
    else:
        raise ValueError(f"Unsupported MIL aggregator type: {args.mil_aggregator_type}")
    
    dir5 = "_".join(arch_setting)
    dir6 = f"exp"
    if args.save_dir_flag is not None:
        dir6 = f"{dir6}_{args.save_dir_flag}"
    dir7 = f"seed{args.seed}"
    dir8 = f"stage_{args.train_stage}"
    args.save_dir = str(Path(args.base_save_dir) / dir1 / dir2 / dir3 / dir4 / dir5 / dir6 / dir7 / dir8)
    logger.info(f"save_dir: {args.save_dir}")

def get_datasets(args):
    """Create datasets that automatically detect clustering paths."""
    indices = load_json(args.data_split_json)["train"]
    train_set = SGMuRCLDataset(
        data_csv=args.data_csv,
        indices=indices,
        num_clusters=args.num_clusters,
        preload=args.preload,
        shuffle=True,
        load_adj_mat=args.graph_encoder_type != "none" or args.mil_aggregator_type == "smtabmil"
    )
    
    # Update num_clusters from dataset if needed
    args.num_clusters = train_set.num_clusters if hasattr(train_set, 'num_clusters') else args.num_clusters
    
    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Total samples: {len(train_set)}")
    logger.info(f"  Patch dimension: {train_set.feature_dim}")
    logger.info(f"  Clusters: {args.num_clusters}")
    
    return train_set, train_set.feature_dim, len(train_set)

def create_model(args, dim_patch):
    """Create model following original MuRCL pattern but with enhanced aggregators."""
    logger.info(f"Creating model {args.mil_aggregator_type} with graph_encoder_type: {args.graph_encoder_type}")
    
    # Step 1: Create Graph Encoder (if using GAT)
    graph_encoder = None
    current_feature_dim = dim_patch

    if args.graph_encoder_type == "gat":
        graph_encoder = BatchedGATWrapper(
            input_dim=dim_patch,
            output_dim=args.gnn_output_dim,
            n_heads=args.gat_heads,
            dropout=args.gnn_dropout
        )
        current_feature_dim = args.gnn_output_dim
        logger.info(f"GAT encoder created. Output dim: {current_feature_dim}")
    else:
        logger.info(f"No graph encoder selected. Using patch dim: {current_feature_dim}")
    
    # Step 2: Create Base MIL Model
    if args.mil_aggregator_type == 'abmil':
        base_model = abmil.ABMIL(
            input_dim=current_feature_dim,
            emb_dim=args.model_dim,
            pool_kwargs={
                'att_dim': args.D,
                'K': getattr(args, "abmil_K", 1),
            },
            ce_criterion=None
        )
    elif args.mil_aggregator_type == 'smtabmil':
        transformer_kwargs = {
            'att_dim': args.D,
            'num_heads': getattr(args, 'transf_num_heads', 8),
            'num_layers': getattr(args, 'transf_num_layers', 2),
            'use_ff': getattr(args, 'transf_use_ff', True),
            'dropout': getattr(args, 'transf_dropout', 0.1),
            'use_sm': getattr(args, 'use_sm_transformer', True),
            'sm_alpha': getattr(args, 'sm_alpha', 0.5),
            'sm_mode': getattr(args, 'sm_mode', 'approx'),
            'sm_steps': getattr(args, 'sm_steps', 10),
        }
        
        pool_kwargs = {
            'att_dim': args.D,
            'sm_alpha': getattr(args, 'sm_alpha', 0.5),
            'sm_mode': getattr(args, 'sm_mode', 'approx'),
            'sm_where': getattr(args, 'sm_where', 'early'),
            'sm_steps': getattr(args, 'sm_steps', 10),
            'sm_spectral_norm': getattr(args, 'sm_spectral_norm', True),
        }
        
        base_model = smtabmil.SMTABMIL(
            input_dim=current_feature_dim,
            emb_dim=args.model_dim,
            transformer_encoder_kwargs=transformer_kwargs,
            pool_kwargs=pool_kwargs,
            ce_criterion=None
        )
    else:
        raise NotImplementedError(f'args.mil_aggregator_type error, {args.mil_aggregator_type}.')

    # Step 3: Create Pipeline (Graph + MIL) if using graph encoder
    if graph_encoder is not None:
        model = GraphAndMILPipeline(
            input_dim=dim_patch,
            graph_encoder=graph_encoder,
            mil_aggregator=base_model
        )
        # For CL wrapping, we need the output dimension of the pipeline
        n_features = args.model_dim
    else:
        model = base_model
        n_features = args.model_dim

    # Step 4: Wrap with CL (Contrastive Learning) following original pattern
    model = cl.CL(model, projection_dim=args.projection_dim, n_features=n_features)
    
    # Step 5: Create FC layer - FIXED TO USE PROJECTION_DIM
    # The FC layer receives the projected features from CL wrapper, not the raw model output
    fc = rlmil.Full_layer(args.projection_dim, args.fc_hidden_dim, args.fc_rnn, args.projection_dim)
    ppo = None

    # Step 6: Load checkpoints based on training stage (following original pattern)
    if args.train_stage == 1:
        pass
    elif args.train_stage == 2:
        if args.checkpoint is None:
            args.checkpoint = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
        assert Path(args.checkpoint).exists(), f"{args.checkpoint} is not exist!"

        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        fc.load_state_dict(checkpoint['fc'])

        state_dim = args.model_dim
        ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.num_clusters)
    elif args.train_stage == 3:
        if args.checkpoint is None:
            args.checkpoint = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
        assert Path(args.checkpoint).exists(), f'{str(args.checkpoint)} is not exists!'

        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        fc.load_state_dict(checkpoint['fc'])

        state_dim = args.model_dim
        ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                        action_std=args.action_std,
                        lr=args.ppo_lr,
                        gamma=args.ppo_gamma,
                        K_epochs=args.K_epochs,
                        action_size=args.num_clusters)
        ppo.policy.load_state_dict(checkpoint['policy'])
        ppo.policy_old.load_state_dict(checkpoint['policy'])
    else:
        raise ValueError

    model = torch.nn.DataParallel(model).cuda()
    fc = fc.cuda()

    assert model is not None, "creating model failed."
    logger.info(f"model Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    logger.info(f"fc Total params: {sum(p.numel() for p in fc.parameters()) / 1e6:.2f}M")
    return model, fc, ppo

def get_optimizer(args, model, fc):
    """Create optimizer with separate learning rates for different components."""
    if args.train_stage != 2:
        params = []
        
        # If using graph encoder, add its parameters with gnn_lr
        if args.graph_encoder_type != "none":
            graph_params = []
            for name, param in model.named_parameters():
                if "graph_encoder" in name:
                    graph_params.append(param)
            if graph_params:
                params.append({"params": graph_params, "lr": args.gnn_lr, "weight_decay": args.gnn_weight_decay})
        
        # All other model parameters (MIL + CL + any other components) with backbone_lr
        other_model_params = []
        for name, param in model.named_parameters():
            if "graph_encoder" not in name or args.graph_encoder_type == "none":
                other_model_params.append(param)
        if other_model_params:
            params.append({"params": other_model_params, "lr": args.backbone_lr, "weight_decay": args.wdecay})
        
        # FC parameters
        params.append({"params": fc.parameters(), "lr": args.fc_lr, "weight_decay": args.wdecay})
        
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params,
                                        lr=0,  # specify in params
                                        momentum=args.momentum,
                                        nesterov=args.nesterov)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, betas=(args.beta1, args.beta2))
        else:
            raise NotImplementedError
    else:
        optimizer = None
        args.epochs = args.ppo_epochs
    return optimizer

def get_scheduler(args, optimizer):
    """Create learning rate scheduler."""
    if optimizer is None:
        return None
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6)
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError
    return scheduler

def get_feats_enhanced(feat_list, cluster_list, adj_list, action_sequence, feat_size, device):
    """Enhanced version of get_feats that handles adjacency matrices."""
    selected_feats, selected_adjs, selected_masks = get_selected_bag_and_graph(
        feat_list, cluster_list, adj_list, action_sequence, feat_size, device
    )
    
    # Stack into batch tensors
    features_batch = torch.stack(selected_feats)  # [B, feat_size, D]
    masks_batch = torch.stack(selected_masks)    # [B, feat_size]
    
    # Handle adjacency matrices
    if any(adj is not None for adj in selected_adjs):
        default_adj = torch.zeros(feat_size, feat_size, device=device, dtype=torch.float32)
        adj_batch = torch.stack([adj if adj is not None else default_adj for adj in selected_adjs])
    else:
        adj_batch = None
    
    return features_batch, adj_batch, masks_batch

def train(args, train_set, model, fc, ppo, criterion, optimizer, scheduler, tb_writer, save_dir):
    """Training function following original MuRCL pattern with enhanced data handling."""
    # Init variables of logging training process
    save_dir = Path(save_dir)
    best_train_loss = BestVariable(order='min')
    header = ['epoch', 'train', 'best_epoch', 'best_train']
    losses_csv = CSVWriter(filename=save_dir / 'losses.csv', header=header)
    results_csv = CSVWriter(filename=save_dir / 'results.csv', header=['epoch', 'final_epoch', 'final_loss'])
    early_stop = EarlyStop(max_num_accordance=args.patience) if args.patience is not None else None

    if args.train_stage == 2:  # stage-2 just training RL module
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()
    memory_list = [rlmil.Memory(), rlmil.Memory()]  # the memory of two branch
    
    for epoch in range(args.epochs):
        logger.info(f"Training Stage: {args.train_stage}, Epoch: {epoch + 1}/{args.epochs}")
        if optimizer is not None:
            current_lrs = [f"group[{k}]: {group['lr']:.1e}" for k, group in enumerate(optimizer.param_groups)]
            logger.info(f"Current LRs: {', '.join(current_lrs)}")

        if hasattr(train_set, 'shuffle'):
            train_set.shuffle()
        length = len(train_set)

        losses = [AverageMeter() for _ in range(args.T)]
        reward_list = [AverageMeter() for _ in range(args.T - 1)]

        progress_bar = tqdm(range(args.num_data))
        feat_list, cluster_list, adj_list, step = [], [], [], 0
        batch_idx = 0

        for data_idx in progress_bar:
            loss_list = []

            # Get data - MATCH STANDARDIZED FORMAT (feat, cluster, label, case_id, adj_mat, coords)
            features, cluster_info, label, case_id, adj_mat, coords = train_set[data_idx % length]
            
            # Validate features before adding to batch
            if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] == 0:
                logger.warning(f"Skipping invalid features for case {case_id}: shape={features.shape if hasattr(features, 'shape') else 'N/A'}")
                continue
            
            # Store data for batch processing
            feat_list.append(features.to(args.device))  # features is already [N, D]
            cluster_list.append(cluster_info)
            adj_list.append(adj_mat.to(args.device) if adj_mat is not None else None)

            step += 1
            if step == args.batch_size:
                try:
                    # Clear GPU cache before processing batch
                    torch.cuda.empty_cache()
                    
                    # First, random choice (following original pattern)
                    action_sequences = [torch.rand((len(feat_list), args.num_clusters), device=args.device) for _ in range(2)]
                    
                    # Enhanced feature selection with adjacency matrices
                    x_views = []
                    for action_seq in action_sequences:
                        features_batch, adj_batch, masks_batch = get_feats_enhanced(
                            feat_list, cluster_list, adj_list, action_seq, args.feat_size, args.device
                        )
                        
                        # Apply mixup following original pattern
                        features_mixed, _, _ = mixup(features_batch, args.alpha)
                        
                        # Create the correct input format for CL wrapper
                        # The CL wrapper expects tuples of (features, adj_mat, mask)
                        x_views.append((features_mixed, adj_batch, masks_batch))

                    if args.train_stage != 2:
                        # Forward pass through CL wrapper -> returns (z_list, states_list)
                        z_list, states = model(x_views)
                        # Apply FC to the projected features
                        outputs = [fc(z, restart=True) for z in z_list]
                    else:  # stage 2 just training RL
                        with torch.no_grad():
                            z_list, states = model(x_views)
                            outputs = [fc(z, restart=True) for z in z_list]

                    # Compute contrastive loss between the two views
                    loss = criterion(outputs[0], outputs[1])
                    loss_list.append(loss)
                    losses[0].update(loss.data.item(), len(feat_list))

                    # Compute similarity for reward calculation
                    similarity_last = torch.cosine_similarity(outputs[0], outputs[1]).view(1, -1)
                    
                    # Multi-step training loop (T steps)
                    for patch_step in range(1, args.T):
                        # Select features by RL(ppo) or random
                        if args.train_stage == 1:  # stage 1 doesn't have module ppo
                            action_sequences = [torch.rand((len(feat_list), args.num_clusters), device=args.device) for _ in range(2)]
                        else:
                            if patch_step == 1:
                                # Create indices by different states and memory for two views
                                action_sequences = [ppo.select_action(s.to(args.device), m, restart_batch=True) for s, m in zip(states, memory_list)]
                            else:
                                action_sequences = [ppo.select_action(s.to(args.device), m) for s, m in zip(states, memory_list)]
                        
                        # Enhanced feature selection with adjacency matrices
                        x_views = []
                        for action_seq in action_sequences:
                            features_batch, adj_batch, masks_batch = get_feats_enhanced(
                                feat_list, cluster_list, adj_list, action_seq, args.feat_size, args.device
                            )
                            
                            # Apply mixup following original pattern
                            features_mixed, _, _ = mixup(features_batch, args.alpha)
                            x_views.append((features_mixed, adj_batch, masks_batch))

                        if args.train_stage != 2:
                            z_list, states = model(x_views)
                            outputs = [fc(z, restart=False) for z in z_list]
                        else:
                            with torch.no_grad():
                                z_list, states = model(x_views)
                                outputs = [fc(z, restart=False) for z in z_list]

                        loss = criterion(outputs[0], outputs[1])
                        loss_list.append(loss)
                        losses[patch_step].update(loss.data.item(), len(feat_list))

                        # Compute reward for PPO
                        similarity = torch.cosine_similarity(outputs[0], outputs[1]).view(1, -1)
                        reward = similarity_last - similarity  # decrease similarity for reward
                        similarity_last = similarity

                        reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                        for m in memory_list:
                            m.rewards.append(reward)

                    # Update models parameters
                    total_loss = sum(loss_list) / args.T
                    if args.train_stage != 2:
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                    else:
                        # PPO update
                        for m in memory_list:
                            ppo.update(m)

                    # Clean temp batch variables
                    for m in memory_list:
                        m.clear_memory()
                    
                    # Clear intermediate tensors
                    del x_views, z_list, states, outputs, loss_list, total_loss
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM error at batch {batch_idx}. Skipping this batch.")
                    # Clear everything and continue
                    for m in memory_list:
                        m.clear_memory()
                    torch.cuda.empty_cache()
                
                feat_list, cluster_list, adj_list, step = [], [], [], 0
                batch_idx += 1

                progress_bar.set_description(
                    f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                    f"Loss: {losses[-1].avg:.4f}. "
                )
                progress_bar.update()
        progress_bar.close()
        
        if scheduler is not None and epoch >= args.warmup:
            scheduler.step()

        train_loss = losses[-1].avg
        # Write to tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar('train/1.train_loss', train_loss, epoch)
            if args.T > 1 and len(reward_list) > 0 and reward_list[-1].count > 0:
                tb_writer.add_scalar('train/reward', reward_list[-1].avg, epoch)

        # Choose the best result
        is_best = best_train_loss.compare(train_loss, epoch + 1, inplace=True)
        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'fc': fc.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }
        save_checkpoint(state, is_best, str(save_dir))
        
        # Logging
        losses_csv.write_row([epoch + 1, train_loss, best_train_loss.epoch, best_train_loss.best])
        results_csv.write_row([epoch + 1, best_train_loss.epoch, best_train_loss.best])
        logger.info(f"Loss: {train_loss:.4f}, Best: {best_train_loss.best:.4f}, Epoch: {best_train_loss.epoch:2}")
        if args.T > 1 and len(reward_list) > 0 and reward_list[-1].count > 0:
            logger.info(f"Avg Reward: {reward_list[-1].avg:.4f}")
        logger.info("-" * 60)

        # Early Stop
        if early_stop is not None:
            early_stop.update(best_train_loss.best)
            if early_stop.is_stop():
                logger.info("Early stopping triggered!")
                break

    if tb_writer is not None:
        tb_writer.close()

def run(args):
    """Main execution function."""
    init_seeds(args.seed)

    if args.save_dir is None:
        create_save_dir(args)
    else:
        args.save_dir = str(Path(args.base_save_dir) / args.save_dir)
    args.save_dir = increment_path(Path(args.save_dir), exist_ok=args.exist_ok, sep='_')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # Dataset
    train_set, dim_patch, train_length = get_datasets(args)
    args.num_data = train_length * args.data_repeat
    args.eval_step = int(args.num_data / args.batch_size)
    logger.info(f"train_length: {train_length}, epoch_step: {args.num_data}, eval_step: {args.eval_step}")

    # Model, Criterion, Optimizer and Scheduler
    model, fc, ppo = create_model(args, dim_patch)
    criterion = NT_Xent(args.batch_size, args.temperature)
    optimizer = get_optimizer(args, model, fc)
    scheduler = get_scheduler(args, optimizer)

    # Save arguments
    with open(Path(args.save_dir) / 'args.yaml', 'w') as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    logger.info(f"Arguments: {vars(args)}")

    # TensorBoard
    tb_writer = SummaryWriter(args.save_dir) if args.use_tensorboard else None

    # Start training
    train(args, train_set, model, fc, ppo, criterion, optimizer, scheduler, tb_writer, args.save_dir)

# Argument parsing functions (keeping the enhanced ones we created)
def add_graph_arguments(parser):
    """Add graph-related command line arguments."""
    parser.add_argument("--graph_encoder_type", type=str, default="none", choices=["none", "gat"],
                        help="Type of graph encoder to use (none, gat)")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256, help="Hidden dimension of the GNN layers [256]")
    parser.add_argument("--gnn_output_dim", type=int, default=256, help="Output dimension of the GNN layers [256]")
    parser.add_argument("--gnn_num_layers", type=int, default=2, help="Number of GNN layers [2]")
    parser.add_argument("--gnn_dropout", type=float, default=0.1, help="Dropout rate for GNN layers [0.1]")
    parser.add_argument("--gat_heads", type=int, default=4, help="Number of heads for GAT layers [4]")
    parser.add_argument("--gnn_lr", type=float, default=0.001, help="Learning rate for GNN layers [0.001]")
    parser.add_argument("--gnn_weight_decay", type=float, default=5e-4, help="Weight decay for GNN layers [5e-4]")

def add_mil_arguments(parser):
    """Add MIL aggregator-related command line arguments."""
    parser.add_argument("--mil_aggregator_type", type=str, default="abmil", choices=["abmil", "smtabmil"],
                        help="Type of MIL aggregator to use (abmil, smtabmil)")
    parser.add_argument("--abmil_K", type=int, default=1, help="Number of attention heads for ABMIL [1]")

def add_dataset_arguments(parser):
    """Add dataset-related command line arguments."""
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters for k-means [10]")

def add_sm_arguments(parser):
    """Add Sm-related arguments for SmTransformerSmABMIL."""
    parser.add_argument("--sm_alpha", type=float, default=0.5, help="Alpha for Sm [0.5]")
    parser.add_argument("--sm_where", type=str, default="early", help="Where to apply Sm (early, late)")
    parser.add_argument("--sm_steps", type=int, default=10, help="Number of steps for Sm [10]")
    parser.add_argument("--transf_num_heads", type=int, default=8, help="Number of transformer heads [8]")
    parser.add_argument("--use_sm_transformer", action="store_true", default=True, help="Use Sm in transformer")
    parser.add_argument("--transf_num_layers", type=int, default=2, help="Number of transformer layers [2]")
    parser.add_argument("--transf_use_ff", action="store_true", default=True, help="Use feedforward in transformer")
    parser.add_argument("--transf_dropout", type=float, default=0.1, help="Transformer dropout rate [0.1]")
    parser.add_argument("--sm_mode", type=str, default="approx", choices=["approx", "exact"], help="Sm mode [approx]")
    parser.add_argument("--sm_spectral_norm", action="store_true", default=True, help="Use spectral norm in Sm")

def main():
    parser = argparse.ArgumentParser()
    # Data (keeping original structure)
    parser.add_argument('--dataset', type=str, default='Camelyon16', help="dataset name")
    parser.add_argument('--data_csv', type=str, default='', help="the .csv filepath used")
    parser.add_argument('--data_split_json', type=str, default='/path/to/data_split.json')
    parser.add_argument('--preload', action='store_true', default=False, help="preload the patch features, default False")
    parser.add_argument('--data_repeat', type=int, default=10, help="contrastive learning need more iteration to train")
    parser.add_argument('--feat_size', default=1024, type=int, help="the size of selected WSI set")
    
    # Add enhanced arguments
    add_dataset_arguments(parser)
    add_graph_arguments(parser)
    add_mil_arguments(parser)
    add_sm_arguments(parser)
    
    # Train (keeping original structure)
    parser.add_argument('--train_stage', default=1, type=int, help="select training stage")
    parser.add_argument('--T', default=6, type=int, help="maximum length of the sequence of RNNs")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help="specify the optimizer used")
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'StepLR', 'CosineAnnealingLR'], help="specify the lr scheduler used")
    parser.add_argument('--batch_size', type=int, default=128, help="the batch size for training")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ppo_epochs', type=int, default=30, help="the training epochs for RL")
    parser.add_argument('--backbone_lr', default=1e-4, type=float, help='the learning rate for MIL encoder')
    parser.add_argument('--fc_lr', default=1e-4, type=float, help='the learning rate for FC')
    parser.add_argument('--temperature', type=float, default=1.0, help="the temperature coefficient of contrastive loss")
    parser.add_argument('--momentum', type=float, default=0.9, help="the momentum of SGD optimizer")
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--warmup', default=0, type=float, help="the number of epoch for training without lr scheduler")
    parser.add_argument('--wdecay', default=1e-5, type=float, help="the weight decay of optimizer")
    parser.add_argument('--patience', type=int, default=None, help="early stopping patience")

    # Architecture (keeping original structure but enhanced)
    parser.add_argument('--checkpoint', default=None, type=str, help="path to the stage-1/2 checkpoint")
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--model_dim', type=int, default=512)
    
    # Architecture - PPO (keeping original)
    parser.add_argument('--policy_hidden_dim', type=int, default=512)
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    
    # Architecture - Full_layer (keeping original)
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--fc_hidden_dim', type=int, default=1024)
    parser.add_argument('--fc_rnn', action='store_true', default=True)
    
    # Architecture - ABMIL (keeping original)
    parser.add_argument('--D', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Save and Global (keeping original)
    parser.add_argument('--use_tensorboard', action='store_true', default=False)
    parser.add_argument('--base_save_dir', type=str, default='./results')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_dir_flag', type=str, default=None)
    parser.add_argument('--exist_ok', action='store_true', default=False)
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=985, help="random state")
    
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    # Pandas print setting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    torch.set_num_threads(8)

    main()