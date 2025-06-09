import torch
import os
import wandb

from utils import get_local_rank, ddp_setup, plot_att_hist
from dataset_loader import load_dataset
from model_builder import build_MIL_model

from matplotlib import pyplot as plt
from evaluate import evaluate
from predict import predict
from Trainer import Trainer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_optimizer_and_scheduler(model, config):

    weight_decay = 0
    print('Using weight_decay =', weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=weight_decay)

    milestone_init = max(int(config.epochs*0.1), 5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, last_epoch=-1, total_iters=milestone_init)        

    return optimizer, scheduler

def train_test(config, run_train=True, run_test=True):

    local_rank = get_local_rank()

    if config.distributed:
        ddp_setup()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Sanity check
    torch.inverse(torch.ones((0, 0), device=device))

    model = None

    if run_train:

        if local_rank == 0:
            print('Starting training...')

        train_dataset, val_dataset = load_dataset(config, mode='train_val')
        test_dataset = load_dataset(config, mode='test')

        input_dim = train_dataset.data_shape[0]
        print('Input dim:', input_dim)

        if config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
            shuffle = True
        
        train_collate_fn = lambda x: train_dataset.collate_fn(x, use_sparse=config.use_sparse)
        val_collate_fn = lambda x: val_dataset.collate_fn(x, use_sparse=config.use_sparse)
        test_collate_fn = lambda x: test_dataset.collate_fn(x, use_sparse=config.use_sparse)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=shuffle, 
            num_workers=config.num_workers, 
            sampler=train_sampler, 
            collate_fn=train_collate_fn, 
            pin_memory=config.pin_memory
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers, 
            sampler=val_sampler, 
            collate_fn=val_collate_fn, 
            pin_memory=config.pin_memory
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers, 
            sampler=test_sampler, 
            collate_fn=test_collate_fn, 
            pin_memory=config.pin_memory
        )

        model = build_MIL_model(input_dim, config)
        print('Model:\n', model)
        print('Number of parameters:', count_parameters(model))

        if config.balance_loss:
            class_counts = train_dataset.get_class_counts()
            pos_weight = torch.FloatTensor([class_counts[0]/class_counts[1]])
            print('Using pos_weight=', pos_weight)
        else:
            pos_weight = None   

        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
        optimizer, scheduler = get_optimizer_and_scheduler(model, config)

        if config.distributed:
            model = model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = model.to(device)

        if local_rank == 0:
            wandb_run = wandb.run
        else:
            wandb_run = None

        trainer = Trainer(
            model, 
            criterion, 
            optimizer, 
            scheduler, 
            device=device, 
            wandb_run=wandb_run, 
            early_stop_patience=config.patience
        )
        trainer.train(config.epochs, train_dataloader, val_dataloader, test_dataloader)
        best_model_state_dict = trainer.get_best_model_state_dict()
        model.load_state_dict(best_model_state_dict)

        if config.distributed:
            torch.distributed.destroy_process_group()

        if local_rank == 0:

            print('Finished training')

            best_model_state_dict = trainer.get_best_model_state_dict()

            if config.save_weights_path is not None:
                if not os.path.exists(os.path.dirname(config.save_weights_path)):
                    os.makedirs(os.path.dirname(config.save_weights_path))
                torch.save(best_model_state_dict, config.save_weights_path)

                if config.use_wandb:
                    wandb_run.save(config.save_weights_path)

                    print('Weights saved in:', config.save_weights_path)

    if local_rank == 0 and run_test:
        print('Starting test...')
        test_dataset = load_dataset(config, mode='test')
        
        input_dim = test_dataset.data_shape[0]
        print('Input dim:', input_dim)

        if model is None:
            model = build_MIL_model(input_dim, config)
            print('Model:\n', model)
            print('Number of parameters:', count_parameters(model))

            if config.load_weights_path is not None:
                if os.path.exists(config.load_weights_path):
                    print('Loading weights from:', config.load_weights_path)
                    weights_dict = torch.load(config.load_weights_path)
                    model.load_state_dict(weights_dict, strict=False)
                else:
                    print(f'Weights not found in: {config.load_weights_path}. Trying to load from wandb...')
                    if wandb.run is not None:
                        weights_file = wandb.run.file('weights/best.pt').download(replace=True, root='/tmp/francastro-team/')
                        weights_dict = torch.load(weights_file.name)
                        model.load_state_dict(weights_dict, strict=True)
        
        if not run_train:
            _, val_dataset = load_dataset(config, mode='train_val')
        
        val_collate_fn = lambda x: val_dataset.collate_fn(x, use_sparse=config.use_sparse)
        test_collate_fn = lambda x: test_dataset.collate_fn(x, use_sparse=config.use_sparse)
                
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=config.num_workers, 
            sampler=None, 
            collate_fn=val_collate_fn
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=config.num_workers, 
            sampler=None, 
            collate_fn=test_collate_fn
        )

        if config.test_in_cpu:
            evaluate_device = 'cpu'
        else:
            evaluate_device = device

        T_true, y_true, T_logits_pred, f_pred, s_pred, bag_idx = predict(model, test_dataloader, evaluate_device)
        _, val_y_true, _, val_f_pred, _, _ = predict(model, val_dataloader, evaluate_device)

        metrics = evaluate(T_true, y_true, T_logits_pred, f_pred, val_y_true, val_f_pred)

        for metric in metrics:
            print('{:<25s}: {:s}'.format(metric, str(metrics[metric])))
        
        print('Plotting attention values distribution')
        
        T_pred = (T_logits_pred > 0).astype(int)
        fig_att_hist = plot_att_hist(f_pred, y_true, T_pred, bag_idx)

        if wandb.run is None:

            if not os.path.exists(config.results_dir):
                os.makedirs(config.results_dir)

            fig_att_hist.savefig(os.path.join(config.results_dir, f'attention_hist.png'))
        else:
            wandb.run.log(metrics)
            wandb.run.log({'attention_hist': wandb.Image(fig_att_hist)})
        
        plt.close(fig_att_hist)

        print('Finished test')