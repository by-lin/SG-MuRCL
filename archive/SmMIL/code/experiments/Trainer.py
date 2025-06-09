import torch
import numpy as np

from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from copy import deepcopy

class Trainer:
    def __init__(
            self, 
            model, 
            criterion, 
            optimizer, 
            lr_scheduler, 
            annealing_scheduler_dict = None,
            device = 'cpu', 
            wandb_run = None, 
            early_stop_patience = None,
            distributed = False
        ):
        self.model = model  
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.annealing_scheduler_dict = annealing_scheduler_dict
        self.device = device
        self.wandb_run = wandb_run
        self.early_stop_patience = early_stop_patience
        self.distributed = distributed

        if self.early_stop_patience is None:
            self.early_stop_patience = np.inf

        self.best_model_state_dict = None
        self.best_auroc = None
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
    
    def _is_main_process(self):
        if self.distributed:
            return self.device.index == 0
        else:
            return True
    
    def _log(self, metrics):
        if self.wandb_run is not None:
            self.wandb_run.log(metrics)
    
    def _get_model_state_dict(self):
        if self.distributed:
            state_dict = deepcopy(self.model.module.state_dict())
        else:
            state_dict = deepcopy(self.model.state_dict())
        return state_dict

    def train(self, max_epochs, train_dataloader, val_dataloader=None, test_dataloader=None):

        if val_dataloader is None:
            val_dataloader = train_dataloader
        
        if self.best_model_state_dict is None:
            self.best_model_state_dict = self._get_model_state_dict()
            self.best_auroc = -np.inf
            self.best_loss = np.inf

        early_stop_count = 0
        disable_pbar = not self._is_main_process()
        for epoch in range(1, max_epochs+1):

            if self.distributed:
                train_dataloader.sampler.set_epoch(epoch)
                val_dataloader.sampler.set_epoch(epoch)

            # Train loop
            train_metrics = self.train_loop(train_dataloader, disable_pbar=disable_pbar, epoch=epoch)
            self._log(train_metrics)
            torch.cuda.empty_cache() # clear cache

            # Validation loop
            val_metrics = self.eval_loop(val_dataloader, disable_pbar=disable_pbar, epoch=epoch)

            self._log(val_metrics)
            torch.cuda.empty_cache() # clear cache
            
            # Test loop
            if test_dataloader is not None:
                test_metrics = self.eval_loop(test_dataloader, disable_pbar=disable_pbar, epoch=epoch, mode='test')
                self._log(test_metrics)
                torch.cuda.empty_cache() # clear cache
            
            if self.lr_scheduler is not None:
                # lr_sch.step(val_metrics['val/auroc'])
                self.lr_scheduler.step()
            
            if self.annealing_scheduler_dict is not None:
                for annealing_scheduler in self.annealing_scheduler_dict.values():
                    annealing_scheduler.step()
                
            if self._is_main_process():
                print(f'Best model AUROC: {self.best_auroc}, Current model AUROC: {val_metrics["val/auroc"]}')
                print(f'Best model Loss: {self.best_loss}, Current model Loss: {val_metrics["val/BCEWithLogitsLoss"]}')
                print(f'Best model AUROC-Loss: {self.best_auroc - self.best_loss}, Current model AUROC-Loss: {val_metrics["val/auroc"] - val_metrics["val/BCEWithLogitsLoss"]}')
                if val_metrics['val/auroc'] - val_metrics['val/BCEWithLogitsLoss'] < self.best_auroc - self.best_loss:
                    early_stop_count += 1
                    print(f'Early stopping count: {early_stop_count}')
                else:
                    self.best_auroc = val_metrics['val/auroc']
                    self.best_loss = val_metrics['val/BCEWithLogitsLoss']
                    self.best_model_state_dict = self._get_model_state_dict()
                    early_stop_count = 0
                
                if early_stop_count >= self.early_stop_patience:
                    print('Reached early stopping condition')
                    break


    def train_loop(self, dataloader, disable_pbar=False, epoch=0):

        if self.annealing_scheduler_dict is not None:
            print(f'[Trainer] (Epoch {epoch}) Loss coefficients:')
            for loss_name in self.annealing_scheduler_dict.keys():
                print(f'Loss: {loss_name}, Coef: {self.annealing_scheduler_dict[loss_name]()}')

        self.model.train()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=disable_pbar)
        pbar.set_description(f"Train - Epoch {epoch}")
        T_list = []
        T_logits_pred_list = []
        sum_loss_dict = { 'loss' : 0.0, self.criterion._get_name() : 0.0 }

        

        for batch_idx, batch in pbar:
            X, T, y, adj_mat, mask = batch   # X: (batch_size, bag_size, 3, 512, 512), 
                                             # T: (batch_size, bag_size), y: (batch_size, 1), 
                                             # adj_mat: sparse coo tensor (batch_size, bag_size, bag_size),
                                             # mask: (batch_size, bag_size)
            X = X.to(self.device)
            T = T.to(self.device)
            adj_mat = adj_mat.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            T_logits_pred, loss_dict = self.model.compute_loss(T_labels=T.float(), X=X, adj_mat=adj_mat, mask=mask)

            loss = 0.0
            for loss_name, loss_value in loss_dict.items():
                coef = 1.0
                if self.annealing_scheduler_dict is not None:
                    if loss_name in self.annealing_scheduler_dict.keys():
                        coef = self.annealing_scheduler_dict[loss_name]()
                loss += coef*loss_value
                if loss_name not in sum_loss_dict.keys():
                    sum_loss_dict[loss_name] = 0.0
                sum_loss_dict[loss_name] += loss_value.item()
            sum_loss_dict['loss'] += loss.item()

            loss.backward()
            self.optimizer.step()

            T_list.append(T.detach().cpu().numpy())
            T_logits_pred_list.append(T_logits_pred.detach().cpu().numpy())

            if batch_idx < (len(dataloader) - 1):
                pbar.set_postfix({f'train/{loss_name}' : sum_loss_dict[loss_name] / (batch_idx + 1) for loss_name in sum_loss_dict})
            else:
                T = np.concatenate(T_list) # (batch_size*bag_size,)
                T_logits_pred = np.concatenate(T_logits_pred_list) # (batch_size*bag_size,)
                
                try:
                    auroc_score = roc_auc_score(T, T_logits_pred)
                except:
                    auroc_score = 0.0

                train_metrics = {f'train/{loss_name}' : sum_loss_dict[loss_name] / (batch_idx + 1) for loss_name in sum_loss_dict}
                train_metrics['train/auroc'] = auroc_score
                pbar.set_postfix(train_metrics)
            
            del X, T, y, adj_mat, mask, T_logits_pred, loss
        pbar.close()
        return train_metrics
    
    def eval_loop(self, dataloader, disable_pbar=False, epoch=0, mode='val'):

        if mode=='val':
            name = 'Validation'
        elif mode=='test':
            name = 'Test'
        else:
            raise ValueError(f'Invalid mode: {mode}')

        self.model.eval()
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=disable_pbar)
        pbar.set_description(f"{name} - Epoch {epoch}")
        T_list = []
        T_logits_pred_list = []
        sum_loss_dict = { 'loss' : 0.0, self.criterion._get_name() : 0.0 }
        with torch.no_grad():
            for batch_idx, batch in pbar:
                X, T, y, adj_mat, mask = batch   # X: (batch_size, bag_size, 3, 512, 512), 
                                                # T: (batch_size, bag_size), y: (batch_size, 1), 
                                                # adj_mat: sparse coo tensor (batch_size, bag_size, bag_size),
                                                # mask: (batch_size, bag_size)
                X = X.to(self.device)
                T = T.to(self.device)
                adj_mat = adj_mat.to(self.device)
                mask = mask.to(self.device)

                T_logits_pred, loss_dict = self.model.compute_loss(T_labels=T.float(), X=X, adj_mat=adj_mat, mask=mask)

                loss = 0.0
                for loss_name, loss_value in loss_dict.items():
                    coef = 1.0
                    if self.annealing_scheduler_dict is not None:
                        if loss_name in self.annealing_scheduler_dict.keys():
                            coef = self.annealing_scheduler_dict[loss_name]()
                    if loss_name not in sum_loss_dict.keys():
                        sum_loss_dict[loss_name] = 0.0
                    sum_loss_dict[loss_name] += loss_value.item()
                    loss += coef*loss_value.item()
                sum_loss_dict['loss'] += loss                

                T_list.append(T.detach().cpu().numpy())
                T_logits_pred_list.append(T_logits_pred.detach().cpu().numpy())

                if batch_idx < (len(dataloader) - 1):
                    # pbar.set_postfix({f'{mode}/loss' : sum_loss / (batch_idx + 1)})
                    pbar.set_postfix({f'{mode}/{loss_name}' : sum_loss_dict[loss_name] / (batch_idx + 1) for loss_name in sum_loss_dict})
                else:
                    T = np.concatenate(T_list) # (batch_size, 1)
                    T_logits_pred = np.concatenate(T_logits_pred_list) # (batch_size, 1)
                    
                    try:
                        auroc_score = roc_auc_score(T, T_logits_pred)
                    except ValueError:
                        auroc_score = 0.0                              
                    
                    # metrics = {f'{mode}/loss' : sum_loss / (batch_idx + 1), f'{mode}/auroc' : auroc_score}
                    metrics = {f'{mode}/{loss_name}' : sum_loss_dict[loss_name] / (batch_idx + 1) for loss_name in sum_loss_dict}
                    metrics[f'{mode}/auroc'] = auroc_score
                    pbar.set_postfix(metrics)
                del X, T, y, adj_mat, mask, T_logits_pred
        pbar.close()        
        return metrics

    
    def get_best_model_state_dict(self):
        return self.best_model_state_dict
