import torch

from models import DeepMILAttModel

def build_MIL_model(input_dim, args, pos_weight=None):

    ce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    if args.model_name == 'abmil':

        return DeepMILAttModel(
            input_dim=input_dim,
            emb_dim=args.emb_dim,
            pool_kwargs={
                'att_dim': args.pool_att_dim,
            },
            ce_criterion=ce_criterion
        )
    elif args.model_name == 'transformer_abmil':

        return DeepMILAttModel(
            input_dim=input_dim,
            emb_dim=args.emb_dim,
            transformer_encoder_kwargs={
                'att_dim': args.transf_att_dim,
                'num_heads': args.transf_num_heads,
                'num_layers': args.transf_num_layers,
                'use_ff': args.transf_use_ff,
                'dropout': args.transf_dropout,
            },
            pool_kwargs={
                'att_dim': args.pool_att_dim,
            },
            ce_criterion=ce_criterion
        )

    elif args.model_name == 'sm_abmil':
        
        return DeepMILAttModel(
            input_dim=input_dim,
            emb_dim=args.emb_dim,
            pool_kwargs={
                'att_dim': args.pool_att_dim,
                'sm_alpha': args.sm_alpha,
                'sm_mode' : args.sm_mode,
                'sm_where' : args.sm_where,
                'sm_steps' : args.sm_steps,
                'sm_spectral_norm' : args.sm_spectral_norm,
            },
            ce_criterion=ce_criterion
        )
    elif args.model_name == 'sm_transformer_abmil':
        return DeepMILAttModel(
            input_dim=input_dim,
            emb_dim=args.emb_dim,
            transformer_encoder_kwargs={
                'att_dim': args.transf_att_dim,
                'num_heads': args.transf_num_heads,
                'num_layers': args.transf_num_layers,
                'use_ff': args.transf_use_ff,
                'dropout': args.transf_dropout,
                'use_sm': args.sm_transformer,
                'sm_alpha': args.sm_alpha,
                'sm_mode': args.sm_mode,
                'sm_steps' : args.sm_steps,
            },
            pool_kwargs={
                'att_dim': args.pool_att_dim,
                'sm_alpha': args.sm_alpha,
                'sm_mode' : args.sm_mode,
                'sm_where' : args.sm_where,
                'sm_steps' : args.sm_steps,
                'sm_spectral_norm' : args.sm_spectral_norm,
            },
            ce_criterion=ce_criterion
        )
    else:
        raise NotImplementedError
