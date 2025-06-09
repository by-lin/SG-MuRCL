import argparse
import yaml

def alpha_type(value):
    if value in ['trainable']:
        return value
    else:
        return float(value)

def correct_args(args):

    if args.model_name in ['abmil', 'transformer_abmil']:
        args.sm_alpha = 0.0
        args.sm_mode = None
        args.sm_steps = 0
        args.sm_where = None
        args.sm_transformer = False
        args.sm_spectral_norm = False
        args.use_inst_distances = False

    if args.sm_alpha in [0.0, None]:
        args.sm_mode = None
        args.sm_where = None
    
    return args

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train_test', type=str, help="Mode to run the code (train/test)")
    parser.add_argument('--use_wandb', action='store_true', help="Use wandb or not")
    parser.add_argument('--wandb_project', default='Sm-NeurIPS24', type=str, help="Wandb project name")
    
    parser.add_argument('--num_workers', default=12, type=int, help="Number of workers to load data")
    parser.add_argument('--pin_memory', action='store_true', help="Pin memory or not")
    parser.add_argument('--distributed', action='store_true', help="Use distributed training")
    parser.add_argument('--test_in_cpu', action='store_true', help="Test in cpu")
    parser.add_argument('--use_sparse', action='store_true', help="Use sparse tensors to store the adjacency matrix")

    # path settings
    parser.add_argument('--weights_dir', default='/work/work_fran/SmoothAttention/weights/', type=str, metavar='PATH', help="Path to save the model weights")   
    parser.add_argument('--results_dir', default='results/', type=str, metavar='PATH', help="Path to save the results") 

    # experiment settings
    parser.add_argument('--seed', type=int, default=0, help="Seed")
    parser.add_argument('--dataset_name', default='rsna-features_resnet18', type=str, help="Dataset to use")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size of training")
    parser.add_argument('--val_prop', type=float, default=0.2, help="Proportion of validation data")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--config_file', type=str, default='/work/work_fran/SmMIL/code/experiments/config.yml', help="Config file to load the settings")

    # model settings
    parser.add_argument('--model_name', type=str, default='abmil', help="Model name")

    # Sm settings
    parser.add_argument('--sm_alpha', type=alpha_type, default='trainable', help="alpha for the Sm operator")
    parser.add_argument('--sm_mode', type=str, default='approx', help="Smooth mode for the Sm operator")
    parser.add_argument('--sm_steps', type=int, default=10, help="Number of steps to approximate the Sm operator")
    parser.add_argument('--sm_where', type=str, default='early', help="Where to place the Sm operator within the attention pool")
    parser.add_argument('--sm_transformer', action='store_true', help="Whether to use the Sm operator in the transformer encoder")
    parser.add_argument('--sm_spectral_norm', action='store_true', help="Use spectral normalization or not")
    parser.add_argument('--use_inst_distances', action='store_true', help="Use instance distances or not to build the adjacency matrix")    

    # training settings
    parser.add_argument('--balance_loss', action='store_true', help="Balance the loss using class weights")
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay for the optimizer")

    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            yaml_dict = yaml.safe_load(f)
    
        for key, value in yaml_dict.items():
            setattr(args, key, value)
    
    args = correct_args(args)    
    
    return args