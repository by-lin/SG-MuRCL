import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import sys
sys.path.append('code/')

import torch
import wandb

from utils import get_local_rank, seed_everything
from arguments import get_arguments
from train_test import train_test

print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

def main():
    config = get_arguments()

    local_rank = get_local_rank()

    if local_rank == 0:
        if config.use_wandb:
            wandb_config = config.__dict__
            for k in wandb_config.keys():
                if wandb_config[k] is None:
                    wandb_config[k] = 'None'
            wandb.init(project=config.wandb_project, config=wandb_config)
            config.save_weights_path = config.load_weights_path = wandb.run.dir + '/weights/best.pt'
        else:
            config.save_weights_path = config.load_weights_path = None

        print('Arguments:')
        for arg in vars(config):
            print('{:<25s}: {:s}'.format(arg, str(getattr(config, arg))))
    
    if config.distributed:
        print('Distributed training!')

    seed_everything(config.seed)

    run_train = 'train' in config.mode
    run_test = 'test' in config.mode

    train_test(config, run_train, run_test)
    
    print('Done!')

if __name__ == "__main__":
    main()
