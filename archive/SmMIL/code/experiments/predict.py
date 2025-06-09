import torch
import numpy as np

from tqdm import tqdm

def adjust_f(f_pred, T_logits_pred):
    """Adjust f_pred based on T_logits_pred

    Args:
        f_pred: (batch_size, bag_size,)
        T_logits_pred: (batch_size,)
    """
    f_pred_list = []
    for i in range(T_logits_pred.shape[0]):
        if T_logits_pred[i] > 0:
            f_pred_list.append(f_pred[i])
        else:
            new_f_pred = -5.0*torch.ones_like(f_pred[i])
            f_pred_list.append(new_f_pred)
    f_pred = torch.stack(f_pred_list)
    return f_pred

def masked_softmax(x, mask, dim=1, epsilon=1e-5):
    """Softmax with mask

    Args:
        x: (batch_size, bag_size)
        mask: (batch_size, bag_size)
    """
    exp_x = torch.exp(x)
    exp_x = exp_x * mask.float()
    sum_exp_x = torch.sum(exp_x, dim=dim) + epsilon
    return exp_x / sum_exp_x

def normalize_s(s_pred, mask):
    """Normalize s_pred

    Args:
        s_pred: (batch_size, bag_size,)
        mask: (batch_size, bag_size)
    """
    s_pred_list = []
    for i in range(s_pred.shape[0]):
        s_pred_i = s_pred[i][mask[i] == 1]
        s_pred_norm = (s_pred_i - torch.min(s_pred_i)) / (torch.max(s_pred_i) - torch.min(s_pred_i) + 1e-5)
        s_pred_list.append(s_pred_norm)
    s_pred = torch.stack(s_pred_list)
    return s_pred

def predict(model, dataloader, device = 'cuda'):

    model = model.to(device)
    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar.set_description("Test")
    
    bag_idx_list = []
    T_list = []
    T_logits_pred_list = []
    y_list = []
    f_pred_list = []
    s_pred_list = []
    # with torch.no_grad():
    for bag_idx, batch in pbar:

        batch_size = batch[0].shape[0]

        if batch_size != 1:
            raise ValueError("[predict] Batch size must be 1")

        X, T, y, adj_mat, mask = batch # X: (batch_size, bag_size, 3, 512, 512), T: (batch_size, bag_size), y: (batch_size, 1), adj_mat: (batch_size, bag_size, bag_size), mask: (batch_size, bag_size)
        X = X.to(device)
        T = T.to(device)
        y = y.to(device)
        adj_mat = adj_mat.to(device)
        mask = mask.to(device)

        T_logits_pred, f_pred = model.predict(X, adj_mat, mask, return_y_pred=True) # T_logits_pred: (batch_size,), f_pred: (batch_size, bag_size)
        
        T_logits_pred = T_logits_pred.detach() # (batch_size,)
        f_pred = f_pred.detach() # (batch_size, bag_size)

        f_pred = adjust_f(f_pred, T_logits_pred) # (batch_size, bag_size)

        s_pred = masked_softmax(f_pred, mask, dim=1) # (batch_size, bag_size)
        s_pred = normalize_s(s_pred, mask) # (batch_size, bag_size)

        f_pred = f_pred.view(-1)[mask.view(-1) == 1] # (batch_size*bag_size,)
        s_pred = s_pred.view(-1)[mask.view(-1) == 1] # (batch_size*bag_size,)
        y = y.view(-1)[mask.view(-1) == 1]

        T_list.append(T.cpu().numpy())
        T_logits_pred_list.append(T_logits_pred.cpu().numpy())
        y_list.append(y.cpu().numpy())
        f_pred_list.append(f_pred.cpu().numpy())
        s_pred_list.append(s_pred.cpu().numpy())
        bag_idx_list.append(np.repeat(bag_idx, len(y)))

    T = np.concatenate(T_list) # (batch_size*bag_size,)
    y = np.concatenate(y_list, axis=0) # (batch_size*bag_size, 1)

    T_logits_pred = np.concatenate(T_logits_pred_list) # (batch_size,)
    T_pred = np.where(T_logits_pred > 0, 1, 0) # (batch_size, 1)
    f_pred = np.concatenate(f_pred_list, axis=0) # (batch_size, 1)
    s_pred = np.concatenate(s_pred_list, axis=0) # (batch_size, 1)
    bag_idx = np.concatenate(bag_idx_list, axis=0) # (batch_size*bag_size,)

    # Discard unlabeled instances
    keep_idx = np.where( (y == 0) | (y == 1) )[0]
    y = y[keep_idx]
    f_pred = f_pred[keep_idx]

    return T, y, T_logits_pred, f_pred, s_pred, bag_idx