import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import copy
import yaml
import argparse
import pandas as pd
from tqdm import tqdm

import logging
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from utils.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, accuracy, init_seeds, \
    load_json, get_metrics, get_score, save_checkpoint
from models import rlmil, abmil, smtabmil

# Updated imports to match our enhanced pipeline
from datasets.datasets import SGMuRCLDataset, get_selected_bag_and_graph
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
    dir2 = "RLMIL"
    
    # RLMIL specific settings
    rlmil_setting = [
        f"T{args.T}",
        f"as{args.action_std}",
        f"pg{args.ppo_gamma}",
        f"phd{args.policy_hidden_dim}",
        f"fhd{args.fc_hidden_dim}",
    ]
    
    # Add graph encoder settings to directory name
    if args.graph_encoder_type != "none":
        rlmil_setting.extend([
            f"gnn{args.graph_encoder_type}",
            f"ghd{args.gnn_hidden_dim}",
            f"god{args.gnn_output_dim}",
            f"gnl{args.gnn_num_layers}",
        ])
        if args.graph_encoder_type == "gat":
            rlmil_setting.append(f"gah{args.gat_heads}")
    
    dir3 = "_".join(rlmil_setting)
    
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
    dir6 = args.train_method
    if args.save_dir_flag is not None:
        dir6 = f"{dir6}_{args.save_dir_flag}"
    dir7 = f"seed{args.seed}"
    dir8 = f"stage_{args.train_stage}"
    
    args.save_dir = str(Path(args.base_save_dir) / dir1 / dir2 / dir3 / dir4 / dir5 / dir6 / dir7 / dir8)
    logger.info(f"save_dir: {args.save_dir}")


def get_datasets(args):
    """Load datasets using our enhanced SGMuRCLDataset."""
    logger.info(f"train_data: {args.train_data}")
    indices = load_json(args.data_split_json)
    
    # Use enhanced SGMuRCLDataset
    train_set = SGMuRCLDataset(
        data_csv=args.data_csv,
        indices=indices[args.train_data],
        num_clusters=args.num_clusters,
        preload=args.preload,
        shuffle=True,
        load_adj_mat=args.graph_encoder_type == "gat"
    )
    valid_set = SGMuRCLDataset(
        data_csv=args.data_csv,
        indices=indices['valid'],
        num_clusters=args.num_clusters,
        preload=args.preload,
        shuffle=False,
        load_adj_mat=args.graph_encoder_type == "gat"
    )
    test_set = SGMuRCLDataset(
        data_csv=args.data_csv,
        indices=indices['test'],
        num_clusters=args.num_clusters,
        preload=args.preload,
        shuffle=False,
        load_adj_mat=args.graph_encoder_type == "gat"
    )
    
    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Total samples: {len(train_set)}")
    logger.info(f"  Patch dimension: {train_set.feature_dim}")
    logger.info(f"  Clusters: {args.num_clusters}")
    
    return {'train': train_set, 'valid': valid_set, 'test': test_set}, train_set.feature_dim, len(train_set)


def create_model(args, dim_patch):
    """Create model compatible with enhanced MuRCL training."""
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
    
    # Step 2: Create Base MIL Model (matching train_MuRCL structure)
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
        args.feature_num = args.model_dim  # Pipeline output is same as MIL output
    else:
        model = base_model
        args.feature_num = args.model_dim
    
    # Step 4: Create FC layer
    fc = rlmil.Full_layer(args.feature_num, args.fc_hidden_dim, args.fc_rnn, args.num_classes)
    ppo = None

    # Step 5: Checkpoint loading logic (adapted for enhanced models)
    if args.train_method in ['finetune', 'linear']:
        if args.train_stage == 1:
            assert args.checkpoint_pretrained is not None and Path(
                args.checkpoint_pretrained).exists(), f"{args.checkpoint_pretrained} is not exists!"

            checkpoint = torch.load(args.checkpoint_pretrained)
            model_state_dict = checkpoint['model_state_dict']
            
            # Handle enhanced model state dict
            for k in list(model_state_dict.keys()):
                if k.startswith('encoder') and not k.startswith('encoder.fc') and not k.startswith(
                        'encoder.classifiers'):
                    model_state_dict[k[len('encoder.'):]] = model_state_dict[k]
                del model_state_dict[k]
            
            msg_model = model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"msg_model missing_keys: {msg_model.missing_keys}")

            if args.train_method == 'linear':
                for n, p in model.named_parameters():
                    if n.startswith('fc') or n.startswith('classifiers') or n.startswith('instance_classifiers'):
                        logger.info(f"not_fixed_key: {n}")
                    else:
                        p.requires_grad = False

        elif args.train_stage == 2:
            if args.checkpoint_stage is None:
                args.checkpoint_stage = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
            assert Path(args.checkpoint_stage).exists(), f"{args.checkpoint_stage} is not exist!"

            checkpoint = torch.load(args.checkpoint_stage)
            model.load_state_dict(checkpoint['model_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            assert args.checkpoint_pretrained is not None and Path(
                args.checkpoint_pretrained).exists(), f"{args.checkpoint_pretrained} is not exists!"
            checkpoint = torch.load(args.checkpoint_pretrained)
            state_dim = args.model_dim
            ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                            action_std=args.action_std,
                            lr=args.ppo_lr,
                            gamma=args.ppo_gamma,
                            K_epochs=args.K_epochs,
                            action_size=args.num_clusters)
            if 'policy' in checkpoint:
                ppo.policy.load_state_dict(checkpoint['policy'])
                ppo.policy_old.load_state_dict(checkpoint['policy'])

        elif args.train_stage == 3:
            if args.checkpoint_stage is None:
                args.checkpoint_stage = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
            assert Path(args.checkpoint_stage).exists(), f"{args.checkpoint_stage} is not exist!"

            checkpoint = torch.load(args.checkpoint_stage)
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

            if args.train_method == 'linear':
                for n, p in model.named_parameters():
                    if n.startswith('fc') or n.startswith('classifiers') or n.startswith('instance_classifiers'):
                        logger.info(f"not_fixed_key: {n}")
                    else:
                        p.requires_grad = False
        else:
            raise ValueError

    elif args.train_method in ['scratch']:
        if args.train_stage == 1:
            pass
        elif args.train_stage == 2:
            if args.checkpoint_stage is None:
                args.checkpoint_stage = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
            assert Path(args.checkpoint_stage).exists(), f"{args.checkpoint_stage} is not exist!"

            checkpoint = torch.load(args.checkpoint_stage)
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
            if args.checkpoint_stage is None:
                args.checkpoint_stage = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
            assert Path(args.checkpoint_stage).exists(), f'{str(args.checkpoint_stage)} is not exists!'

            checkpoint = torch.load(args.checkpoint_stage)
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
    else:
        raise ValueError

    model = torch.nn.DataParallel(model).cuda()
    fc = fc.cuda()

    assert model is not None, "creating model failed."
    logger.info(f"model Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    logger.info(f"fc Total params: {sum(p.numel() for p in fc.parameters()) / 1e6:.2f}M")
    return model, fc, ppo


def get_criterion(args):
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError(f"args.loss error, error value is {args.loss}.")
    return criterion


def get_optimizer(args, model, fc):
    """Create optimizer with separate learning rates for different components."""
    if args.train_stage != 2:
        params = []
        
        # If using graph encoder, add its parameters with gnn_lr
        if args.graph_encoder_type == "gat":
            graph_params = []
            for name, param in model.named_parameters():
                if "graph_encoder" in name:
                    graph_params.append(param)
            if graph_params:
                params.append({"params": graph_params, "lr": args.gnn_lr, "weight_decay": args.gnn_weight_decay})
        
        # All other model parameters (MIL + any other components) with backbone_lr
        other_model_params = []
        for name, param in model.named_parameters():
            if args.graph_encoder_type != "gat" or "graph_encoder" not in name:
                other_model_params.append(param)
        if other_model_params:
            params.append({"params": other_model_params, "lr": args.backbone_lr, "weight_decay": args.wdecay})
        
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


# Train Model Functions---------------------------------------------------------------------------------------------
def train_model(args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler):
    """Unified training function for both ABMIL and SMTABMIL models."""
    logger.info(f"train_{args.mil_aggregator_type.upper()} (Epoch {epoch + 1}/{args.epochs})...")
    length = len(train_set)
    train_set.shuffle()

    losses_records = [AverageMeter() for _ in range(args.T)]
    top1_records = [AverageMeter() for _ in range(args.T)]
    reward_records = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()

    progress_bar = tqdm(range(args.num_data), desc=f"Epoch {epoch+1} Training")
    
    # Lists to accumulate data for a batch
    feat_list_orig, cluster_list_orig, label_list_orig, adj_list_orig = [], [], [], []
    step_in_batch = 0
    
    # For overall epoch metrics
    all_epoch_labels, all_epoch_outputs = [], []

    for data_idx in progress_bar:
        current_loss_sequence = []

        # Get data - MATCH STANDARDIZED FORMAT (feat, cluster, label, case_id, adj_mat, coords)
        features, cluster_info, label, case_id, adj_mat, coords = train_set[data_idx % length]

        feat_list_orig.append(features.to(args.device))
        cluster_list_orig.append(cluster_info)
        label_list_orig.append(label.to(args.device))
        if args.graph_encoder_type == "gat":
            adj_list_orig.append(adj_mat.to(args.device) if adj_mat is not None else None)
        else:
            adj_list_orig.append(None)

        step_in_batch += 1

        if step_in_batch == args.batch_size or data_idx == args.num_data - 1:
            batch_labels = torch.stack(label_list_orig)

            # Initial action (random for first step)
            current_action_sequence = torch.rand((len(feat_list_orig), args.num_clusters), device=args.device)

            selected_features_batched, selected_adj_mats_list, selected_masks_list = get_selected_bag_and_graph(
                feat_list_orig, cluster_list_orig, adj_list_orig,
                current_action_sequence, args.feat_size, args.device
            )
            
            ppo_input_states = None
            if args.train_stage != 2:
                # Forward through enhanced model
                bag_embeddings, _ = model(selected_features_batched, adj_mat=selected_adj_mats_list, mask=selected_masks_list)
                final_logits = fc(bag_embeddings, restart=True)
                ppo_input_states = bag_embeddings.detach()
            else:
                with torch.no_grad():
                    bag_embeddings, _ = model(selected_features_batched, adj_mat=selected_adj_mats_list, mask=selected_masks_list)
                    final_logits = fc(bag_embeddings, restart=True)
                    ppo_input_states = bag_embeddings

            loss = criterion(final_logits, batch_labels)
            current_loss_sequence.append(loss)
            losses_records[0].update(loss.item(), len(feat_list_orig))
            acc = accuracy(final_logits, batch_labels, topk=(1,))[0]
            top1_records[0].update(acc.item(), len(feat_list_orig))

            confidence_last = torch.gather(F.softmax(final_logits.detach(), 1), dim=1, index=batch_labels.view(-1, 1)).view(1, -1)

            # Multi-step training loop
            for patch_step in range(1, args.T):
                if args.train_stage == 1:
                    current_action_sequence = torch.rand((len(feat_list_orig), args.num_clusters), device=args.device)
                else:
                    if ppo_input_states is None:
                        logger.error("ppo_input_states is None in PPO loop. This should not happen.")
                        ppo_input_states = torch.zeros(len(feat_list_orig), args.model_dim, device=args.device)

                    current_action_sequence = ppo.select_action(
                        ppo_input_states.to(args.device),
                        memory, 
                        restart_batch=(patch_step == 1)
                    )
                
                selected_features_batched, selected_adj_mats_list, selected_masks_list = get_selected_bag_and_graph(
                    feat_list_orig, cluster_list_orig, adj_list_orig,
                    current_action_sequence, args.feat_size, args.device
                )

                if args.train_stage != 2:
                    bag_embeddings, _ = model(selected_features_batched, adj_mat=selected_adj_mats_list, mask=selected_masks_list)
                    final_logits = fc(bag_embeddings, restart=False)
                    ppo_input_states = bag_embeddings.detach()
                else:
                    with torch.no_grad():
                        bag_embeddings, _ = model(selected_features_batched, adj_mat=selected_adj_mats_list, mask=selected_masks_list)
                        final_logits = fc(bag_embeddings, restart=False)
                        ppo_input_states = bag_embeddings
                
                loss = criterion(final_logits, batch_labels)
                current_loss_sequence.append(loss)
                losses_records[patch_step].update(loss.item(), len(feat_list_orig))
                acc = accuracy(final_logits, batch_labels, topk=(1,))[0]
                top1_records[patch_step].update(acc.item(), len(feat_list_orig))

                confidence = torch.gather(F.softmax(final_logits.detach(), 1), dim=1, index=batch_labels.view(-1, 1)).view(1, -1)
                reward = confidence - confidence_last
                confidence_last = confidence
                reward_records[patch_step - 1].update(reward.mean().item(), len(feat_list_orig))
                memory.rewards.append(reward)

            # Optimization
            total_loss_for_batch = sum(current_loss_sequence) / args.T
            if args.train_stage != 2:
                optimizer.zero_grad()
                total_loss_for_batch.backward()
                optimizer.step()
            
            if args.train_stage != 1:
                ppo.update(memory)
            
            memory.clear_memory()
            torch.cuda.empty_cache()

            # Store results for epoch metrics
            all_epoch_labels.append(batch_labels.cpu())
            all_epoch_outputs.append(final_logits.detach().cpu())

            # Reset lists for the next batch
            feat_list_orig, cluster_list_orig, label_list_orig, adj_list_orig = [], [], [], []
            step_in_batch = 0
            
            progress_bar.set_description(
                f"E:{epoch+1} B:{data_idx // args.batch_size +1}/{args.eval_step} Loss:{losses_records[-1].avg:.4f} Acc:{top1_records[-1].avg:.4f}"
            )
            progress_bar.update()

    progress_bar.close()
    if args.train_stage != 2 and scheduler is not None and epoch >= args.warmup:
        scheduler.step()

    # Calculate overall epoch metrics
    if all_epoch_labels and all_epoch_outputs:
        final_epoch_labels = torch.cat(all_epoch_labels)
        final_epoch_outputs = torch.cat(all_epoch_outputs)
        epoch_acc, epoch_auc, epoch_precision, epoch_recall, epoch_f1_score = get_metrics(final_epoch_outputs, final_epoch_labels)
    else:
        epoch_acc, epoch_auc, epoch_precision, epoch_recall, epoch_f1_score = 0.0, 0.0, 0.0, 0.0, 0.0

    return losses_records[-1].avg, epoch_acc, epoch_auc, epoch_precision, epoch_recall, epoch_f1_score


def test_model(args, test_set, model, fc, ppo, memory, criterion):
    """Unified test function for both ABMIL and SMTABMIL models."""
    logger.info(f"test_{args.mil_aggregator_type.upper()}...")
    losses_records = [AverageMeter() for _ in range(args.T)]
    reward_records = [AverageMeter() for _ in range(args.T - 1)]

    model.eval()
    fc.eval()
    if ppo: 
        ppo.policy_old.eval()

    all_test_outputs, all_test_labels, all_test_case_ids = [], [], []
    feat_list_orig, cluster_list_orig, label_list_orig, adj_list_orig, case_id_list_batch = [], [], [], [], []
    
    # Iterate through the entire test set once
    for data_idx, data_item in enumerate(tqdm(test_set, desc="Testing")):
        current_loss_sequence = []

        # Get data - MATCH STANDARDIZED FORMAT
        features, cluster_info, label, case_id, adj_mat, coords = data_item
        
        feat_list_orig.append(features.to(args.device))
        cluster_list_orig.append(cluster_info)
        label_list_orig.append(label.to(args.device))
        case_id_list_batch.append(case_id)
        if args.graph_encoder_type == "gat":
            adj_list_orig.append(adj_mat.to(args.device) if adj_mat is not None else None)
        else:
            adj_list_orig.append(None)

        # Process if batch_size is reached or it's the last item
        if len(feat_list_orig) == args.batch_size or data_idx == len(test_set) - 1:
            batch_labels = torch.stack(label_list_orig)

            with torch.no_grad():
                # Initial action (random for testing consistency)
                current_action_sequence = torch.rand((len(feat_list_orig), args.num_clusters), device=args.device)

                selected_features_batched, selected_adj_mats_list, selected_masks_list = get_selected_bag_and_graph(
                    feat_list_orig, cluster_list_orig, adj_list_orig,
                    current_action_sequence, args.feat_size, args.device
                )
                
                ppo_input_states = None
                bag_embeddings, _ = model(selected_features_batched, adj_mat=selected_adj_mats_list, mask=selected_masks_list)
                final_logits = fc(bag_embeddings, restart=True)
                ppo_input_states = bag_embeddings

                loss = criterion(final_logits, batch_labels)
                current_loss_sequence.append(loss)
                losses_records[0].update(loss.item(), len(feat_list_orig))
                
                confidence_last = torch.gather(F.softmax(final_logits, 1), dim=1, index=batch_labels.view(-1, 1)).view(1, -1)

                for patch_step in range(1, args.T):
                    if args.train_stage == 1 or not ppo:
                        current_action_sequence = torch.rand((len(feat_list_orig), args.num_clusters), device=args.device)
                    else:
                        current_action_sequence = ppo.select_action(
                            ppo_input_states.to(args.device), 
                            memory,
                            restart_batch=(patch_step == 1),
                            is_eval=True
                        )

                    selected_features_batched, selected_adj_mats_list, selected_masks_list = get_selected_bag_and_graph(
                        feat_list_orig, cluster_list_orig, adj_list_orig,
                        current_action_sequence, args.feat_size, args.device
                    )

                    bag_embeddings, _ = model(selected_features_batched, adj_mat=selected_adj_mats_list, mask=selected_masks_list)
                    final_logits = fc(bag_embeddings, restart=False)
                    ppo_input_states = bag_embeddings
                    
                    loss = criterion(final_logits, batch_labels)
                    current_loss_sequence.append(loss)
                    losses_records[patch_step].update(loss.item(), len(feat_list_orig))

                    confidence = torch.gather(F.softmax(final_logits, 1), dim=1, index=batch_labels.view(-1, 1)).view(1, -1)
                    reward = confidence - confidence_last
                    confidence_last = confidence
                    reward_records[patch_step - 1].update(reward.mean().item(), len(feat_list_orig))
                
                all_test_outputs.append(final_logits.cpu())
                all_test_labels.append(batch_labels.cpu())
                all_test_case_ids.extend(case_id_list_batch)

            # Reset lists for the next batch
            feat_list_orig, cluster_list_orig, label_list_orig, adj_list_orig, case_id_list_batch = [], [], [], [], []

    final_test_outputs = torch.cat(all_test_outputs)
    final_test_labels = torch.cat(all_test_labels)
    
    acc, auc, precision, recall, f1_score = get_metrics(final_test_outputs, final_test_labels)

    return losses_records[-1].avg, acc, auc, precision, recall, f1_score, final_test_outputs, final_test_labels, all_test_case_ids


# Basic Functions---------------------------------------------------------------------------------------------------
def train(args, train_set, valid_set, test_set, model, fc, ppo, memory, criterion, optimizer, scheduler, tb_writer):
    # Init variables
    save_dir = args.save_dir
    best_train_acc = BestVariable(order='max')
    best_valid_acc = BestVariable(order='max')
    best_test_acc = BestVariable(order='max')
    best_train_auc = BestVariable(order='max')
    best_valid_auc = BestVariable(order='max')
    best_test_auc = BestVariable(order='max')
    best_train_loss = BestVariable(order='min')
    best_valid_loss = BestVariable(order='min')
    best_test_loss = BestVariable(order='min')
    best_score = BestVariable(order='max')
    final_loss, final_acc, final_auc, final_precision, final_recall, final_f1_score, final_epoch = 0., 0., 0., 0., 0., 0., 0
    header = ['epoch', 'train', 'valid', 'test', 'best_train', 'best_valid', 'best_test']
    losses_csv = CSVWriter(filename=Path(save_dir) / 'losses.csv', header=header)
    accs_csv = CSVWriter(filename=Path(save_dir) / 'accs.csv', header=header)
    aucs_csv = CSVWriter(filename=Path(save_dir) / 'aucs.csv', header=header)
    results_csv = CSVWriter(filename=Path(save_dir) / 'results.csv',
                            header=['epoch', 'final_epoch', 'final_loss', 'final_acc', 'final_auc', 'final_precision',
                                    'final_recall', 'final_f1_score'])

    best_model = copy.deepcopy({'state_dict': model.state_dict()})
    early_stop = EarlyStop(max_num_accordance=args.patience) if args.patience is not None else None

    for epoch in range(args.epochs):
        print(f"Training Stage: {args.train_stage}, lr:")
        if optimizer is not None:
            for k, group in enumerate(optimizer.param_groups):
                print(f"group[{k}]: {group['lr']}")

        train_loss, train_acc, train_auc, train_precision, train_recall, train_f1_score = \
            train_model(args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler)
        valid_loss, valid_acc, valid_auc, valid_precision, valid_recall, valid_f1_score, *_ = \
            test_model(args, valid_set, model, fc, ppo, memory, criterion)
        test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score, *_ = \
            test_model(args, test_set, model, fc, ppo, memory, criterion)

        # Write to tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar('train/1.train_loss', train_loss, epoch)
            tb_writer.add_scalar('test/2.test_loss', valid_loss, epoch)

        # Choose the best result
        if args.picked_method == 'acc':
            is_best = best_valid_acc.compare(valid_acc)
        elif args.picked_method == 'loss':
            is_best = best_valid_loss.compare(valid_loss)
        elif args.picked_method == 'auc':
            is_best = best_valid_auc.compare(valid_auc)
        elif args.picked_method == 'score':
            score = get_score(valid_acc, valid_auc, valid_precision, valid_recall, valid_f1_score)
            is_best = best_score.compare(score, epoch + 1, inplace=True)
        else:
            raise ValueError(f"picked_method error. ")
        if is_best:
            final_epoch = epoch + 1
            final_loss = test_loss
            final_acc = test_acc
            final_auc = test_auc

        # Compute best result
        best_train_acc.compare(train_acc, epoch + 1, inplace=True)
        best_valid_acc.compare(valid_acc, epoch + 1, inplace=True)
        best_test_acc.compare(test_acc, epoch + 1, inplace=True)
        best_train_loss.compare(train_loss, epoch + 1, inplace=True)
        best_valid_loss.compare(valid_loss, epoch + 1, inplace=True)
        best_test_loss.compare(test_loss, epoch + 1, inplace=True)
        best_train_auc.compare(train_auc, epoch + 1, inplace=True)
        best_valid_auc.compare(valid_auc, epoch + 1, inplace=True)
        best_test_auc.compare(test_auc, epoch + 1, inplace=True)

        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'fc': fc.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }
        if is_best:
            best_model = copy.deepcopy(state)
            if args.save_model:
                save_checkpoint(state, is_best, str(save_dir))

        # Save
        losses_csv.write_row([epoch + 1, train_loss, valid_loss, test_loss,
                              (best_train_loss.best, best_train_loss.epoch),
                              (best_valid_loss.best, best_valid_loss.epoch),
                              (best_test_loss.best, best_test_loss.epoch)])
        accs_csv.write_row([epoch + 1, train_acc, valid_acc, test_acc,
                            (best_train_acc.best, best_train_acc.epoch),
                            (best_valid_acc.best, best_valid_acc.epoch),
                            (best_test_acc.best, best_test_acc.epoch)])
        aucs_csv.write_row([epoch + 1, train_auc, valid_auc, test_auc,
                            (best_train_auc.best, best_train_auc.epoch),
                            (best_valid_auc.best, best_valid_auc.epoch),
                            (best_test_auc.best, best_test_auc.epoch)])
        results_csv.write_row(
            [epoch + 1, final_epoch, test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score])

        print(
            f"Train acc: {train_acc:.4f}, Best: {best_train_acc.best:.4f}, Epoch: {best_train_acc.epoch:2}, "
            f"AUC: {train_auc:.4f}, Best: {best_train_auc.best:.4f}, Epoch: {best_train_auc.epoch:2}, "
            f"Loss: {train_loss:.4f}, Best: {best_train_loss.best:.4f}, Epoch: {best_train_loss.epoch:2}\n"
            f"Valid acc: {valid_acc:.4f}, Best: {best_valid_acc.best:.4f}, Epoch: {best_valid_acc.epoch:2}, "
            f"AUC: {valid_auc:.4f}, Best: {best_valid_auc.best:.4f}, Epoch: {best_valid_auc.epoch:2}, "
            f"Loss: {valid_loss:.4f}, Best: {best_valid_loss.best:.4f}, Epoch: {best_valid_loss.epoch:2}\n"
            f"Test  acc: {test_acc:.4f}, Best: {best_test_acc.best:.4f}, Epoch: {best_test_acc.epoch:2}, "
            f"AUC: {test_auc:.4f}, Best: {best_test_auc.best:.4f}, Epoch: {best_test_auc.epoch:2}, "
            f"Loss: {test_loss:.4f}, Best: {best_test_loss.best:.4f}, Epoch: {best_test_loss.epoch:2}\n"
            f"Final Epoch: {final_epoch:2}, Final acc: {final_acc:.4f}, Final AUC: {final_auc:.4f}, Final Loss: {final_loss:.4f}\n"
        )

        # Early Stop
        if early_stop is not None:
            early_stop.update((best_valid_loss.best, best_valid_acc.best, best_valid_auc.best))
            if early_stop.is_stop():
                break

    if tb_writer is not None:
        tb_writer.close()

    return best_model


def test(args, test_set, model, fc, ppo, memory, criterion):
    model.eval()
    fc.eval()
    with torch.no_grad():
        loss, acc, auc, precision, recall, f1_score, outputs_tensor, labels_tensor, case_id_list = \
            test_model(args, test_set, model, fc, ppo, memory, criterion)
        prob = torch.softmax(outputs_tensor, dim=1)
        _, pred = torch.max(prob, dim=1)
        preds = pd.DataFrame(columns=['label', 'pred', 'correct', *[f'prob{i}' for i in range(prob.shape[1])]])
        for i in range(len(case_id_list)):
            preds.loc[case_id_list[i]] = [
                labels_tensor[i].item(),
                pred[i].item(),
                labels_tensor[i].item() == pred[i].item(),
                *[prob[i][j].item() for j in range(prob.shape[1])],
            ]
        preds.index.rename('case_id', inplace=True)

    return loss, acc, auc, precision, recall, f1_score, preds


def run(args):
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
    datasets, dim_patch, train_length = get_datasets(args)
    args.num_data = train_length
    args.eval_step = int(args.num_data / args.batch_size)
    print(f"train_length: {train_length}, epoch_step: {args.num_data}, eval_step: {args.eval_step}")

    # Model, Criterion, Optimizer and Scheduler
    model, fc, ppo = create_model(args, dim_patch)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model, fc)
    scheduler = get_scheduler(args, optimizer)

    # Save arguments
    with open(Path(args.save_dir) / 'args.yaml', 'w') as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    print(args, '\n')

    # TensorBoard
    tb_writer = SummaryWriter(args.save_dir) if args.use_tensorboard else None

    # Start training
    memory = rlmil.Memory()
    best_model = train(args, datasets['train'], datasets['valid'], datasets['test'], model, fc, ppo, memory, criterion,
                       optimizer, scheduler, tb_writer)
    model.module.load_state_dict(best_model['model_state_dict'])
    fc.load_state_dict(best_model['fc'])
    if ppo is not None:
        ppo.policy.load_state_dict(best_model['policy'])
    loss, acc, auc, precision, recall, f1_score, preds = \
        test(args, datasets['test'], model, fc, ppo, memory, criterion)

    # Save results
    preds.to_csv(str(Path(args.save_dir) / 'pred.csv'))
    final_res = pd.DataFrame(columns=['loss', 'acc', 'auc', 'precision', 'recall', 'f1_score'])
    final_res.loc[f'seed{args.seed}'] = [loss, acc, auc, precision, recall, f1_score]
    final_res.to_csv(str(Path(args.save_dir) / 'final_res.csv'))
    print(f'{final_res}\nPredicted Ending.\n')


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='CAMELYON16',
                        help="dataset name")
    parser.add_argument('--data_csv', type=str, default='',
                        help="the .csv filepath used")
    parser.add_argument('--data_split_json', type=str, help='/path/to/data_split.json')
    parser.add_argument('--train_data', type=str, default='train', choices=['train', 'train_sub_per10'],
                        help="specify how much data used")
    parser.add_argument('--preload', action='store_true', default=False,
                        help="preload the patch features, default False")
    parser.add_argument('--feat_size', default=1024, type=int,
                        help="the size of selected WSI set")
    parser.add_argument('--num_clusters', default=10, type=int,
                        help="number of patch clusters")
    
    # Graph Encoder Arguments
    parser.add_argument('--graph_encoder_type', type=str, default='none', choices=['none', 'gat'],
                        help="type of graph encoder to use")
    parser.add_argument('--gnn_hidden_dim', type=int, default=256,
                        help="hidden dimension for GNN")
    parser.add_argument('--gnn_output_dim', type=int, default=256,
                        help="output dimension for GNN")
    parser.add_argument('--gnn_num_layers', type=int, default=2,
                        help="number of GNN layers")
    parser.add_argument('--gnn_dropout', type=float, default=0.1,
                        help="dropout for GNN")
    parser.add_argument('--gat_heads', type=int, default=4,
                        help="number of attention heads for GAT")
    parser.add_argument('--gnn_lr', type=float, default=0.001,
                        help="learning rate for GNN")
    parser.add_argument('--gnn_weight_decay', type=float, default=5e-4,
                        help="weight decay for GNN")
    
    # MIL Aggregator Arguments
    parser.add_argument('--mil_aggregator_type', type=str, default='abmil', choices=['abmil', 'smtabmil'],
                        help="type of MIL aggregator to use")
    parser.add_argument('--abmil_K', type=int, default=1,
                        help="number of attention heads for ABMIL")
    
    # SmMIL Arguments
    parser.add_argument('--sm_alpha', type=float, default=0.5,
                        help="alpha parameter for SmMIL")
    parser.add_argument('--sm_where', type=str, default='early', choices=['early', 'late'],
                        help="where to apply SmMIL")
    parser.add_argument('--sm_steps', type=int, default=10,
                        help="number of steps for SmMIL")
    parser.add_argument('--sm_mode', type=str, default='approx', choices=['exact', 'approx'],
                        help="SmMIL computation mode")
    parser.add_argument('--sm_spectral_norm', action='store_true', default=True,
                        help="use spectral normalization in SmMIL")
    
    # Transformer Arguments
    parser.add_argument('--transf_num_heads', type=int, default=8,
                        help="number of transformer attention heads")
    parser.add_argument('--transf_num_layers', type=int, default=2,
                        help="number of transformer layers")
    parser.add_argument('--transf_use_ff', action='store_true', default=True,
                        help="use feed-forward in transformer")
    parser.add_argument('--transf_dropout', type=float, default=0.1,
                        help="transformer dropout")
    parser.add_argument('--use_sm_transformer', action='store_true', default=True,
                        help="use SmMIL in transformer")
    
    # Train
    parser.add_argument('--train_method', type=str, default='scratch', choices=['scratch', 'finetune', 'linear'])
    parser.add_argument('--train_stage', default=1, type=int,
                        help="select training stage")
    parser.add_argument('--T', default=6, type=int,
                        help="maximum length of the sequence of RNNs")
    parser.add_argument('--checkpoint_stage', default=None, type=str,
                        help="path to the stage-1/2 checkpoint")
    parser.add_argument('--checkpoint_pretrained', type=str, default=None,
                        help='path to the pretrained checkpoint')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help="specify the optimizer used")
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'StepLR', 'CosineAnnealingLR'],
                        help="specify the lr scheduler used")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="the batch size for training")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help="the training epochs for RL")
    parser.add_argument('--backbone_lr', default=1e-4, type=float)
    parser.add_argument('--fc_lr', default=1e-4, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--warmup', default=0, type=float,
                        help="the number of epoch for training without lr scheduler")
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help="the weight decay of optimizer")
    parser.add_argument('--picked_method', type=str, default='score',
                        help="the metric of pick best model from validation dataset")
    parser.add_argument('--patience', type=int, default=None,
                        help="early stopping patience")

    # Architecture
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_dim', type=int, default=512)
    
    # Architecture - PPO
    parser.add_argument('--policy_hidden_dim', type=int, default=512)
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    
    # Architecture - Full_layer
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--fc_hidden_dim', type=int, default=1024)
    parser.add_argument('--fc_rnn', action='store_true', default=True)
    
    # Architecture - ABMIL/SMTABMIL
    parser.add_argument('--D', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Loss
    parser.add_argument('--loss', default='CrossEntropyLoss', type=str, choices=['CrossEntropyLoss'],
                        help='loss name')
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help="use TensorBoard")
    
    # Save
    parser.add_argument('--base_save_dir', type=str, default='./results')
    parser.add_argument('--save_dir', type=str, default=None,
                        help="specify the save directory")
    parser.add_argument('--save_dir_flag', type=str, default=None,
                        help="append a string to the end of save_dir")
    parser.add_argument('--exist_ok', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    
    # Global
    parser.add_argument('--device', default='2',
                        help='cuda device')
    parser.add_argument('--seed', type=int, default=985)
    
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    # Pandas print setting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    torch.set_num_threads(8)

    main()