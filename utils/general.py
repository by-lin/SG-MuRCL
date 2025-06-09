import os
import re
import csv
import yaml
import json
import glob
import shutil
import random
import logging
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def init_seeds(seed=0):
    # # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    files = glob.glob('./**/' + file, recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"File Not Found: {file}")
    if len(files) > 1:
        raise RuntimeError(f"Multiple files match '{file}', specify exact path: {files}")
    return files[0]

def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"{re.escape(path.stem)}{sep}(\d+)$", Path(d).stem) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"

def load_yaml(filename):
    with open(filename) as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)

def dump_json(data_dict, filename):
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(data_dict, fp)

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        return json.load(fp)

class EarlyStop:
    def __init__(self, max_num_accordance=5):
        self.max_num_accordance = max_num_accordance
        self.base_variable = ()
        self.num_accordance = 0

    def update(self, variable):
        if variable == self.base_variable:
            self.num_accordance += 1
        else:
            self.num_accordance = 1
            self.base_variable = variable

    def is_stop(self):
        return self.num_accordance >= self.max_num_accordance

class CSVWriter:
    def __init__(self, filename, header=None, sep=',', append=False):
        self.filename = filename
        self.sep = sep
        if Path(self.filename).exists() and not append:
            os.remove(self.filename)
        if header is not None:
            self.write_row(header)

    def write_row(self, row):
        try:
            with open(self.filename, 'a+') as fp:
                csv_writer = csv.writer(fp, delimiter=self.sep)
                csv_writer.writerow(row)
        except IOError as e:
            logger.error(f"Failed to write row to CSV {self.filename}: {e}")

    def write_rows(self, rows):
        try:
            with open(self.filename, 'a+') as fp:
                csv_writer = csv.writer(fp, delimiter=self.sep)
                csv_writer.writerows(rows)
        except IOError as e:
            logger.error(f"Failed to write rows to CSV {self.filename}: {e}")

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BestVariable:
    def __init__(self, order='max'):
        self.order = order
        self.reset()

    def reset(self):
        self.best = float('-inf') if self.order == 'max' else float('inf')
        self.epoch = 0

    def compare(self, val, epoch=None, inplace=False):
        flag = (val > self.best if self.order == 'max' else val < self.best)
        if flag:
            if inplace:
                if epoch is None:
                    logger.warning("BestVariable.compare(): inplace=True but epoch=None")
                self.best = val
                if epoch is not None:
                    self.epoch = epoch
        return flag

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_metrics(outputs, targets):
    """
    Compute evaluation metrics: Accuracy, AUC, Precision, Recall, F1.

    Args:
        outputs (Tensor): Model outputs (logits), shape [B, C]
        targets (Tensor): Ground truth labels, shape [B]

    Returns:
        Tuple: (accuracy, auc, precision, recall, f1_score)
    """
    with torch.no_grad():
        assert outputs.shape[0] == targets.shape[0]
        bs = targets.shape[0]
        num_classes = outputs.shape[1]
        is_multiclass = num_classes > 2

        # Accuracy
        _, preds = torch.max(outputs, dim=1)
        acc = (preds.eq(targets).sum().item()) / bs

        # AUC
        targets_np = targets.cpu().numpy()
        probs_np = torch.softmax(outputs, dim=1).cpu().numpy()
        try:
            if is_multiclass:
                auc = roc_auc_score(targets_np, probs_np, multi_class='ovr')
            else:
                auc = roc_auc_score(targets_np, probs_np[:, 1])
        except ValueError as e:
            logger.warning(f"AUC computation failed: {e}")
            auc = float('nan')

        # Precision, Recall, F1
        try:
            preds_np = preds.cpu().numpy()
            average = 'macro' if is_multiclass else 'binary'
            precision, recall, f1_score, _ = precision_recall_fscore_support(targets_np, preds_np, average=average)
        except ValueError as e:
            logger.warning(f"PRF computation failed: {e}")
            precision = recall = f1_score = float('nan')

    return acc, auc, precision, recall, f1_score

def get_score(acc, auc, precision, recall, f1_score):
    return 0.3 * acc + 0.3 * auc + 0.1 * precision + 0.1 * recall + 0.2 * f1_score

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_path)
        logger.info(f"Saved best model to: {best_path}")


def extract_sub_adjacency_matrix(full_adj_mat, selected_indices):
    """
    Extracts a sub-adjacency matrix corresponding to selected_indices.

    Args:
        full_adj_mat (torch.Tensor or np.ndarray): The full adjacency matrix [N, N].
        selected_indices (torch.Tensor or np.ndarray or list): 1D array/list of selected indices.

    Returns:
        torch.Tensor: The sub-adjacency matrix [k, k] where k is len(selected_indices),
                      or None if inputs are invalid.
    """
    if full_adj_mat is None or selected_indices is None:
        return None
    
    if isinstance(selected_indices, list):
        selected_indices = np.array(selected_indices)

    if isinstance(selected_indices, np.ndarray):
        selected_indices = torch.from_numpy(selected_indices).long()
    
    if not isinstance(selected_indices, torch.Tensor):
        raise TypeError(f"selected_indices must be a list, np.ndarray, or torch.Tensor, got {type(selected_indices)}")

    selected_indices = selected_indices.long().flatten()

    if selected_indices.numel() == 0:
        return None # Or an empty tensor: torch.empty((0,0), device=full_adj_mat.device if isinstance(full_adj_mat, torch.Tensor) else None)

    if not isinstance(full_adj_mat, torch.Tensor):
        # Try to convert if it's a NumPy array, otherwise, it's an unsupported type
        if isinstance(full_adj_mat, np.ndarray):
            full_adj_mat = torch.from_numpy(full_adj_mat)
        else:
            raise TypeError(f"full_adj_mat must be a torch.Tensor or np.ndarray, got {type(full_adj_mat)}")

    # Ensure indices are within bounds
    if selected_indices.max() >= full_adj_mat.shape[0] or selected_indices.min() < 0:
        # This can happen if selected_indices contains -1 (dummy)
        # Filter out invalid indices (e.g., -1 used as a placeholder)
        valid_mask = (selected_indices >= 0) & (selected_indices < full_adj_mat.shape[0])
        selected_indices = selected_indices[valid_mask]
        if selected_indices.numel() == 0:
            return None 
            
    # Advanced indexing to get the submatrix
    # sub_matrix = full_adj_mat[selected_indices, :][:, selected_indices]
    # A more robust way for PyTorch indexing:
    idx = selected_indices.unsqueeze(0) # Shape [1, k]
    sub_matrix = full_adj_mat.index_select(0, selected_indices).index_select(1, selected_indices)
    
    return sub_matrix.float() # Ensure it's float for further processing