# Add this at the very top, before any other imports:
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Go up from run/ to SG-MuRCL/
sys.path.insert(0, str(project_root))

import os
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import logging

from datasets.datasets import WSIWithCluster, mixup, get_selected_bag_and_graph
from utils.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, init_seeds, save_checkpoint, load_json
from models import rlmil, abmil, cl, smtabmil
from utils.losses import NT_Xent

# Add these imports for GAT and pipeline
from models.graph_encoders import GATEncoder, BatchedGATWrapper
from models.pipeline_modules import GraphAndMILPipeline

# --- Logger Setup ---
def setup_logger():
    """Sets up a basic console logger for the script."""
    logging.basicConfig(
        level=logging.INFO, # Log messages at INFO level and above (WARNING, ERROR, CRITICAL)
        format="[%(asctime)s] %(levelname)s - %(message)s", # Log message format
        datefmt="%Y-%m-%d %H:%M:%S" # Timestamp format
    )
    return logging.getLogger(__name__)

logger = setup_logger() # Global logger instance

def create_save_dir(args):
    """
    Create directory to save experiment results by global arguments.
    :param args: the global arguments
    """
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
    
    train_set = WSIWithCluster(
        data_csv=args.data_csv,  # This is all we need!
        indices=indices,
        graph_level=args.graph_level,
        num_patch_clusters=args.num_clusters,
        preload=args.preload,
        shuffle=True,
        patch_random=False,
        load_adj_mat=args.graph_encoder_type != "none"
    )
    
    # Update num_clusters based on actual data if needed
    if hasattr(train_set, "num_patch_clusters"):
        args.num_clusters = train_set.num_patch_clusters
    
    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Total samples: {len(train_set)}")
    logger.info(f"  Patch dimension: {train_set.patch_dim}")
    logger.info(f"  Clusters: {args.num_clusters}")
    
    return train_set, train_set.patch_dim, len(train_set)


def create_model(args, dim_patch_initial):
    """Create the complete SG-MuRCL model pipeline."""
    logger.info(f"Creating model with graph_encoder_type: {args.graph_encoder_type}, mil_aggregator_type: {args.mil_aggregator_type}")
    
    # Step 1: Create Graph Encoder (GAT)
    graph_encoder = None
    current_feature_dim = dim_patch_initial

    if args.graph_encoder_type == "gat":
        # Create GAT encoder
        gat = GATEncoder(
            input_dim=dim_patch_initial,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            num_layers=args.gnn_num_layers,
            heads=args.gat_heads,
            dropout=args.gnn_dropout,
            concat_heads=False  # Average heads for consistent output dimension
        )
        # Wrap with batched processing
        graph_encoder = BatchedGATWrapper(gat)
        current_feature_dim = args.gnn_output_dim  # GAT output dimension
        logger.info(f"Using GAT Encoder. Output dim for MIL: {current_feature_dim}")
    else:
        logger.info(f"No graph encoder selected. Using initial patch dim for MIL: {current_feature_dim}")


    # Step 2: Create MIL Aggregator
    mil_aggregator = None

    if args.mil_aggregator_type == "abmil":
        mil_aggregator = abmil.ABMIL(
            dim_in=current_feature_dim,
            L=args.model_dim,
            D=args.D,
            K=getattr(args, "abmil_K", 1),
            dropout=args.dropout
        )
        logger.info(f"Using ABMIL aggregator with input dim {current_feature_dim}, L={args.model_dim}, D={args.D}")
    
    elif args.mil_aggregator_type == "smtabmil":
        mil_aggregator = smtabmil.SmTransformerSmABMIL(
            dim_in=current_feature_dim,
            L=args.model_dim,
            D=args.D,
            dropout=args.dropout,
            sm_alpha=args.sm_alpha,
            sm_where=args.sm_where,
            sm_steps=args.sm_steps,
            transf_num_heads=args.transf_num_heads,
            use_sm_transformer=getattr(args, "use_sm_transformer", True),
            transf_num_layers=getattr(args, "transf_num_layers", 2),
            transf_use_ff=getattr(args, "transf_use_ff", True),
            transf_dropout=getattr(args, "transf_dropout", 0.1),
            sm_mode=getattr(args, "sm_mode", "approx"),
            sm_spectral_norm=getattr(args, "sm_spectral_norm", False)
        )
        logger.info(f"Using SmTransformerSmABMIL aggregator with input dim {current_feature_dim}, L={args.model_dim}, D={args.D}")
    
    else:
        raise ValueError(f"Unsupported MIL aggregator type: {args.mil_aggregator_type}")

    # Step 3: Create Pipeline (Graph + MIL)
    pipeline_encoder = GraphAndMILPipeline(
        input_dim=dim_patch_initial,
        graph_encoder=graph_encoder,
        mil_aggregator=mil_aggregator
    )

    # Step 4: Wrap with CL (Contrastive Learning)
    model = cl.CL(
        encoder=pipeline_encoder,
        projection_dim=args.projection_dim,
        n_features=args.model_dim  # L, the bag embedding size from MIL aggregator
    )
    logger.info(f"CL wrapper configured with n_features (bag embedding dim L): {args.model_dim}")

    # Step 5: Create FC layer for PPO
    fc = rlmil.Full_layer(
        feature_num=args.feature_num,
        hidden_state_dim=args.fc_hidden_dim,
        fc_rnn=args.fc_rnn,
        class_num=args.projection_dim
    )

    ppo = None

    # Step 6: Load checkpoints based on training stage
    if args.train_stage == 1:
        pass  # No checkpoint loading for stage 1
    elif args.train_stage == 2:
        # Load from stage 1
        if args.checkpoint is None:
            args.checkpoint = str(Path(args.save_dir).parent / "stage_1" / "model_best.pth.tar")
        assert Path(args.checkpoint).exists(), f"{args.checkpoint} does not exist!"

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        fc.load_state_dict(checkpoint["fc"])

        state_dim = args.model_dim
        ppo = rlmil.PPO(
            dim_patch_initial, state_dim, args.policy_hidden_dim, args.policy_conv,
            action_std=args.action_std,
            lr=args.ppo_lr,
            gamma=args.ppo_gamma,
            K_epochs=args.K_epochs,
            action_size=args.num_clusters
        )
    elif args.train_stage == 3:
        # Load from stage 2
        if args.checkpoint is None:
            args.checkpoint = str(Path(args.save_dir).parent / "stage_2" / "model_best.pth.tar")
        assert Path(args.checkpoint).exists(), f"{args.checkpoint} does not exist!"

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        fc.load_state_dict(checkpoint["fc"])

        state_dim = args.model_dim
        ppo = rlmil.PPO(
            dim_patch_initial, state_dim, args.policy_hidden_dim, args.policy_conv,
            action_std=args.action_std,
            lr=args.ppo_lr,
            gamma=args.ppo_gamma,
            K_epochs=args.K_epochs,
            action_size=args.num_clusters
        )
        ppo.policy.load_state_dict(checkpoint["policy"])
        ppo.policy_old.load_state_dict(checkpoint["policy"])
    else:
        raise ValueError(f"Invalid train_stage: {args.train_stage}")

    # Step 7: Move to GPU
    if torch.cuda.is_available() and not args.device == 'cpu':
        model = torch.nn.DataParallel(model)
        model = model.to(args.device)
        fc = fc.to(args.device)
        if ppo is not None:
            ppo = ppo.to(args.device)

    assert model is not None, "Model creation failed"
    logger.info(f"Model Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    logger.info(f"FC Total params: {sum(p.numel() for p in fc.parameters()) / 1e6:.2f}M")
    
    return model, fc, ppo


def get_optimizer(args, model, fc):
    """Create optimizer with separate learning rates for different components."""
    if args.train_stage != 2:
        # Updated to use separate learning rates for different components like GMIL
        params = []
        
        # If using graph encoder, add its parameters with gnn_lr
        if args.graph_encoder_type != "none":
            graph_params = []
            for name, param in model.named_parameters():
                if "graph_encoder" in name:
                    graph_params.append(param)
            if graph_params:
                params.append({"params": graph_params, "lr": args.gnn_lr, "weight_decay": args.gnn_weight_decay})
        
        # MIL aggregator parameters with mil_lr
        mil_params = []
        for name, param in model.named_parameters():
            if "mil_aggregator" in name:
                mil_params.append(param)
        if mil_params:
            params.append({"params": mil_params, "lr": args.mil_lr, "weight_decay": args.mil_weight_decay})
        
        # Other model parameters (CL wrapper, etc.) with backbone_lr
        other_params = []
        for name, param in model.named_parameters():
            if "graph_encoder" not in name and "mil_aggregator" not in name:
                other_params.append(param)
        if other_params:
            params.append({"params": other_params, "lr": args.backbone_lr, "weight_decay": args.wdecay})
        
        # FC parameters
        params.append({"params": fc.parameters(), "lr": args.fc_lr, "weight_decay": args.wdecay})
        
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                params,
                lr=0,  # specified in params
                momentum=args.momentum,
                nesterov=args.nesterov
            )
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                params, 
                betas=(args.beta1, args.beta2)
            )
        else:
            raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")
    else:
        optimizer = None
        args.epochs = args.ppo_epochs
    return optimizer


def get_scheduler(args, optimizer):
    """Create learning rate scheduler."""
    if optimizer is None:
        return None
    if args.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6
        )
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"Scheduler {args.scheduler} not implemented")
    return scheduler


def train(args, train_set, model, fc, ppo, criterion, optimizer, scheduler, tb_writer, save_dir):
    """Training loop updated to handle clustering data and graph processing."""
    # Init variables of logging training process
    save_dir = Path(save_dir)
    best_train_loss = BestVariable(order="min")
    header = ["epoch", "train", "best_epoch", "best_train"]
    losses_csv = CSVWriter(filename=save_dir / "losses.csv", header=header)
    results_csv = CSVWriter(filename=save_dir / "results.csv", header=["epoch", "final_epoch", "final_loss"])
    early_stop = EarlyStop(max_num_accordance=args.patience) if args.patience is not None else None

    if args.train_stage == 2:  # stage-2 just training RL module
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()
    
    memory_list = [rlmil.Memory(), rlmil.Memory()]  # memory for two views
    
    for epoch in range(args.epochs):
        logger.info(f"Training Stage: {args.train_stage}, lr:")
        if optimizer is not None:
            for k, group in enumerate(optimizer.param_groups):
                logger.info(f"group[{k}]: {group['lr']}")

        train_set.shuffle()
        length = len(train_set)

        losses = [AverageMeter() for _ in range(args.T)]
        reward_list = [AverageMeter() for _ in range(args.T - 1)]

        progress_bar = tqdm(range(args.num_data))
        feat_list, cluster_list, adj_mat_list, step = [], [], [], 0
        batch_idx = 0

        for data_idx in progress_bar:
            loss_list = []

            # Get data from dataset - UPDATED to handle WSIWithCluster format
            data_item = train_set[data_idx % length]

            feat, cluster, adj_mat, coords, label, case_id = data_item
            
            # A WSI features is a tensor of shape (num_patches, dim_features)
            assert len(feat.shape) == 2, f"Expected 2D features, got {feat.shape}"
            feat = feat.unsqueeze(0).to(args.device)
            
            # Move adjacency matrix to device if it exists
            if adj_mat is not None:
                adj_mat = adj_mat.to(args.device)

            feat_list.append(feat)
            cluster_list.append(cluster)
            adj_mat_list.append(adj_mat)

            step += 1
            if step == args.batch_size:
                # First step: random action
                action_sequences = [
                    torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device) 
                    for _ in range(2)
                ]
                
                # Get selected bags and adjacency matrices for both views
                selected_bags_and_adj_mats = [
                    get_selected_bag_and_graph(
                        feat_list, cluster_list, adj_mat_list, action_sequence, args.feat_size, args.device
                    ) 
                    for action_sequence in action_sequences
                ]
                
                # Extract bags and adjacency matrices
                x_views = [item[0] for item in selected_bags_and_adj_mats]  # batched selected bags
                adj_mats_views = [item[1] for item in selected_bags_and_adj_mats]  # lists of adj matrices
                
                # Apply mixup
                x_views = [mixup(x, args.alpha)[0] for x in x_views]
                
                if args.train_stage != 2:
                    # Forward through model (CL wrapper -> GraphAndMILPipeline)
                    # Convert batched tensors to lists for pipeline
                    x_views_list = []
                    for x_view in x_views:
                        x_views_list.append([x_view[i] for i in range(x_view.size(0))])
                    
                    outputs, states = model(x_views_list, adj_mats=adj_mats_views)
                    outputs = [fc(o, restart=True) for o in outputs]
                else:  # stage 2 just training RL
                    with torch.no_grad():
                        x_views_list = []
                        for x_view in x_views:
                            x_views_list.append([x_view[i] for i in range(x_view.size(0))])
                        
                        outputs, states = model(x_views_list, adj_mats=adj_mats_views)
                        outputs = [fc(o, restart=True) for o in outputs]

                loss = criterion(outputs[0], outputs[1])
                loss_list.append(loss)
                losses[0].update(loss.data.item(), len(feat_list))

                similarity_last = torch.cosine_similarity(outputs[0], outputs[1]).view(1, -1)
                
                # Multi-step PPO selection
                for patch_step in range(1, args.T):
                    # Select features by PPO
                    if args.train_stage == 1:  # stage 1 doesn"t have PPO
                        action_sequences = [
                            torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
                            for _ in range(2)
                        ]
                    else:
                        if patch_step == 1:
                            # Create actions by different states and memory for two views
                            action_sequences = [
                                ppo.select_action(s.to(0), m, restart_batch=True) 
                                for s, m in zip(states, memory_list)
                            ]
                        else:
                            action_sequences = [
                                ppo.select_action(s.to(0), m) 
                                for s, m in zip(states, memory_list)
                            ]
                    
                    # Get selected bags and adjacency matrices
                    selected_bags_and_adj_mats = [
                        get_selected_bag_and_graph(
                            feat_list, cluster_list, adj_mat_list, action_sequence, args.feat_size, args.device
                        ) 
                        for action_sequence in action_sequences
                    ]
                    
                    x_views = [item[0] for item in selected_bags_and_adj_mats]
                    adj_mats_views = [item[1] for item in selected_bags_and_adj_mats]
                    x_views = [mixup(x, args.alpha)[0] for x in x_views]

                    if args.train_stage != 2:
                        # Convert to list format for pipeline
                        x_views_list = []
                        for x_view in x_views:
                            x_views_list.append([x_view[i] for i in range(x_view.size(0))])
                        
                        outputs, states = model(x_views_list, adj_mats=adj_mats_views)
                        outputs = [fc(o, restart=False) for o in outputs]
                    else:
                        with torch.no_grad():
                            x_views_list = []
                            for x_view in x_views:
                                x_views_list.append([x_view[i] for i in range(x_view.size(0))])
                            
                            outputs, states = model(x_views_list, adj_mats=adj_mats_views)
                            outputs = [fc(o, restart=False) for o in outputs]

                    loss = criterion(outputs[0], outputs[1])
                    loss_list.append(loss)
                    losses[patch_step].update(loss.data.item(), len(feat_list))

                    similarity = torch.cosine_similarity(outputs[0], outputs[1]).view(1, -1)
                    reward = similarity_last - similarity  # decrease similarity for reward
                    similarity_last = similarity

                    reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                    for m in memory_list:
                        m.rewards.append(reward)

                # Update model parameters
                loss = sum(loss_list) / args.T
                if args.train_stage != 2:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    for m in memory_list:
                        ppo.update(m)

                # Clean temp batch variables
                for m in memory_list:
                    m.clear_memory()
                feat_list, cluster_list, adj_mat_list, step = [], [], [], 0
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
            tb_writer.add_scalar("train/1.train_loss", train_loss, epoch)

        # Choose the best result
        is_best = best_train_loss.compare(train_loss, epoch + 1, inplace=True)
        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "fc": fc.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "ppo_optimizer": ppo.optimizer.state_dict() if ppo else None,
            "policy": ppo.policy.state_dict() if ppo else None,
        }
        save_checkpoint(state, is_best, str(save_dir))
        
        # Logging
        losses_csv.write_row([epoch + 1, train_loss, best_train_loss.epoch, best_train_loss.best])
        results_csv.write_row([epoch + 1, best_train_loss.epoch, best_train_loss.best])
        logger.info(f"Loss: {train_loss:.4f}, Best: {best_train_loss.best:.4f}, Epoch: {best_train_loss.epoch:2}\n")

        # Early Stop
        if early_stop is not None:
            early_stop.update(best_train_loss.best)
            if early_stop.is_stop():
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
    args.save_dir = increment_path(Path(args.save_dir), exist_ok=args.exist_ok, sep="_")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if not args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")

    # Dataset
    train_set, dim_patch, train_length = get_datasets(args)
    args.num_data = train_length * args.data_repeat
    args.eval_step = max(1,int(args.num_data / args.batch_size))
    logger.info(f"train_length: {train_length}, epoch_step: {args.num_data}, eval_step: {args.eval_step}")

    # Model, Criterion, Optimizer and Scheduler
    model, fc, ppo = create_model(args, dim_patch)
    criterion = NT_Xent(args.batch_size, args.temperature)
    optimizer = get_optimizer(args, model, fc)
    scheduler = get_scheduler(args, optimizer)

    # Save arguments
    with open(Path(args.save_dir) / "args.yaml", "w") as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    logger.info(f"Arguments: {vars(args)}")

    # TensorBoard
    tb_writer = SummaryWriter(args.save_dir) if args.use_tensorboard else None

    # Start training
    train(args, train_set, model, fc, ppo, criterion, optimizer, scheduler, tb_writer, args.save_dir)


def add_graph_arguments(parser):
    """Add graph-related command line arguments."""
    parser.add_argument("--graph_encoder_type", type=str, default="none", choices=["none", "gat"],
                        help="Type of graph encoder to use (none, gat)")
    # Updated defaults to match GMIL values
    parser.add_argument("--gnn_hidden_dim", type=int, default=256,
                        help="Hidden dimension of the GNN layers [256]")
    parser.add_argument("--gnn_output_dim", type=int, default=256,
                        help="Output dimension of the GNN layers [256]")
    parser.add_argument("--gnn_num_layers", type=int, default=2,
                        help="Number of GNN layers [2]")
    parser.add_argument("--gnn_dropout", type=float, default=0.1,
                        help="Dropout rate for GNN layers [0.1]")
    parser.add_argument("--gat_heads", type=int, default=4,
                        help="Number of attention heads for GAT layers [4]")
    # Add GMIL-style learning rates and weight decay
    parser.add_argument("--gnn_lr", type=float, default=0.001,
                        help="Learning rate for GNN layers [0.001]")
    parser.add_argument("--gnn_weight_decay", type=float, default=5e-4,
                        help="Weight decay for GNN layers [5e-4]")

def add_mil_arguments(parser):
    """Add MIL aggregator-related command line arguments."""
    parser.add_argument("--mil_aggregator_type", type=str, default="abmil", choices=["abmil", "smtabmil"],
                        help="Type of MIL aggregator to use (abmil, smtabmil)")
    parser.add_argument("--abmil_K", type=int, default=1, help="Number of attention heads for ABMIL [1]")
    # Add GMIL-style MIL learning rate and weight decay
    parser.add_argument("--mil_lr", type=float, default=0.0001,
                        help="Learning rate for MIL layers [0.0001]")
    parser.add_argument("--mil_weight_decay", type=float, default=1e-4,
                        help="Weight decay for MIL layers [1e-4]")

def add_dataset_arguments(parser):
    """Add dataset-related command line arguments."""
    parser.add_argument("--graph_level", type=str, default="patch", choices=["patch", "region"],
                        help="Level of graph processing (patch or region)")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters for k-means")
    # Add input dimension to match GMIL
    parser.add_argument("--input_dim", type=int, default=1024,
                        help="Dimension of the input feature size [1024]")

def add_sm_arguments(parser):
    """Add Sm-related arguments for SmTransformerSmABMIL."""
    parser.add_argument("--sm_alpha", type=float, default=0.5, help="Alpha for Sm [0.5]")
    parser.add_argument("--sm_where", type=str, default="early",
                        help="Where to apply Sm (early, late)")
    parser.add_argument("--sm_steps", type=int, default=10, help="Number of steps for Sm [10]")
    parser.add_argument("--transf_num_heads", type=int, default=8, help="Number of transformer heads [8]")
    parser.add_argument("--use_sm_transformer", action="store_true", help="Use Sm in transformer")
    parser.add_argument("--transf_num_layers", type=int, default=2, help="Number of transformer layers [2]")
    parser.add_argument("--transf_use_ff", action="store_true", default=True, help="Use feedforward in transformer")
    parser.add_argument("--transf_dropout", type=float, default=0.1, help="Transformer dropout rate [0.1]")
    parser.add_argument("--sm_mode", type=str, default="approx", choices=["approx", "exact"], help="Sm mode")
    parser.add_argument("--sm_spectral_norm", action="store_true", default=True, help="Use spectral norm in Sm")


def main():
    parser = argparse.ArgumentParser(description="Train MuRCL with GAT and flexible MIL aggregators")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="Camelyon16", help="Dataset name")
    parser.add_argument("--data_csv", type=str, default="", help="The .csv filepath used")
    parser.add_argument("--data_split_json", type=str, default="/path/to/data_split.json")
    parser.add_argument("--preload", action="store_true", default=False,
                        help="Preload the patch features, default False")
    parser.add_argument("--data_repeat", type=int, default=10,
                        help="Contrastive learning needs more iterations to train")
    parser.add_argument("--feat_size", default=1024, type=int,
                        help="The size of selected WSI set (recommend 1024 at 20x magnification)")
    
    # Add modular arguments
    add_dataset_arguments(parser)
    add_graph_arguments(parser)
    add_mil_arguments(parser)
    add_sm_arguments(parser)
    
    # Training arguments
    parser.add_argument("--train_stage", default=1, type=int,
                        help="Select training stage: 1=warm-up, 2=RL, 3=finetune")
    parser.add_argument("--T", default=6, type=int,
                        help="Maximum length of the sequence of RNNs")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"],
                        help="Specify the optimizer used")
    parser.add_argument("--scheduler", type=str, default=None, choices=[None, "StepLR", "CosineAnnealingLR"],
                        help="Specify the lr scheduler used")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--ppo_epochs", type=int, default=30, help="Training epochs for RL")
    parser.add_argument("--backbone_lr", default=1e-4, type=float, help="Learning rate for MIL encoder")
    parser.add_argument("--fc_lr", default=1e-4, type=float, help="Learning rate for FC")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature coefficient of contrastive loss")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum of SGD optimizer")
    parser.add_argument("--nesterov", action="store_true", default=True)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--warmup", default=0, type=float,
                        help="Number of epochs for training without lr scheduler")
    parser.add_argument("--wdecay", default=1e-5, type=float, help="Weight decay of optimizer")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience [10]")
    parser.add_argument("--min_delta", type=float, default=1e-5,
                        help="Min delta for early stopping [1e-5]")

    # Architecture arguments
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Path to the stage-1/2 checkpoint (for training stage-2/3)")
    parser.add_argument("--alpha", type=float, default=0.9, help="Mixup alpha")
    parser.add_argument("--projection_dim", type=int, default=128, help="Projection dimension for contrastive learning")
    parser.add_argument("--model_dim", type=int, default=512, help="Model dimension (L)")
    
    # PPO arguments
    parser.add_argument("--policy_hidden_dim", type=int, default=512)
    parser.add_argument("--policy_conv", action="store_true", default=False)
    parser.add_argument("--action_std", type=float, default=0.5)
    parser.add_argument("--ppo_lr", type=float, default=0.00001)
    parser.add_argument("--ppo_gamma", type=float, default=0.1)
    parser.add_argument("--K_epochs", type=int, default=3)
    
    # Full layer arguments
    parser.add_argument("--feature_num", type=int, default=512)
    parser.add_argument("--fc_hidden_dim", type=int, default=1024)
    parser.add_argument("--fc_rnn", action="store_true", default=True)
    
    # ABMIL arguments
    parser.add_argument("--D", type=int, default=128, help="Intermediate dimension for attention")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    
    # GMIL-style additional arguments
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes [2]")
    parser.add_argument("--class_weighting", action="store_true", default=True, help="Use class weighting")
    parser.add_argument("--average", action="store_true", default=True, 
                        help="Average the score of max-pooling and bag aggregating")
    parser.add_argument("--non_linearity", type=float, default=0.0, 
                        help="Additional nonlinear operation [0.0]")
    
    # Logging arguments
    parser.add_argument("--use_tensorboard", action="store_true", default=False)
    
    # Save arguments
    parser.add_argument("--base_save_dir", type=str, default="./results")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Specify the save directory to save experiment results")
    parser.add_argument("--save_dir_flag", type=str, default=None,
                        help="Append a string to the end of save_dir")
    parser.add_argument("--exist_ok", action="store_true", default=False)
    
    # Global arguments
    parser.add_argument("--device", default="3", help="CUDA device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--seed", type=int, default=985, help="Random state [985]")

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    # Pandas print setting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    torch.set_num_threads(8)

    main()