import torch
import numpy as np


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc

def auprc(labels, preds, pos_label=1):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
    return auc(recall, precision)

def fpr_at_thr_tpr(preds, labels, pos_label=1, thr=0.8):
    """Return the FPR when TPR is at minimum thr.
        
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < thr):
        # No threshold allows TPR >= thr
        return 0
    elif all(tpr >= thr):
        # All thresholds allow TPR >= thr, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= thr]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == thr
        return np.interp(thr, tpr, fpr)

def compute_optimal_thr(y_pred, y):
    """Compute optimal threshold for y_pred

    Args:
        y_pred: (batch_size,)
        y: (batch_size,)
    """
    fpr, tpr, thr = roc_curve(y, y_pred)
    # idx = np.argmax(tpr - fpr, axis=0)
    idx = np.argmax(tpr*(1-fpr), axis=0)
    optimal_thr = thr[idx]
    return optimal_thr

def evaluate(T_true, y_true, T_logits_pred, f_pred, val_y_true=None, val_f_pred=None):

    T_pred = np.where(T_logits_pred > 0, 1, 0) # (batch_size, 1)

    metrics = {}
    metrics['bag/bce_loss'] = torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(T_logits_pred).float(), torch.from_numpy(T_true).float()).item()
    try: 
        metrics['bag/auroc'] = roc_auc_score(T_true, T_logits_pred)
    except ValueError:
        metrics['bag/auroc'] = 0.0
    metrics['bag/auprc'] = auprc(T_true, T_logits_pred)
    # metrics['bag/fpr80'] = fpr_at_thr_tpr(T_logits_pred, T_true, thr=0.80)
    metrics['bag/fpr90'] = fpr_at_thr_tpr(T_logits_pred, T_true, thr=0.90)

    metrics['bag/acc'] = accuracy_score(T_true, T_pred)
    metrics['bag/prec'] = precision_score(T_true, T_pred, zero_division=0)
    metrics['bag/rec'] = recall_score(T_true, T_pred, zero_division=0)
    metrics['bag/f1'] = f1_score(T_true, T_pred, zero_division=0)
    
    metrics[f'inst/auroc'] = roc_auc_score(y_true, f_pred)
    metrics[f'inst/auprc'] = auprc(y_true, f_pred)
    metrics[f'inst/fpr90'] = fpr_at_thr_tpr(f_pred, y_true, thr=0.90)

    y_pred = np.where(f_pred > 0.0, 1, 0)
    metrics[f'inst/acc'] = accuracy_score(y_true, y_pred)
    metrics[f'inst/prec'] = precision_score(y_true, y_pred, zero_division=0)
    metrics[f'inst/rec'] = recall_score(y_true, y_pred, zero_division=0)
    metrics[f'inst/f1'] = f1_score(y_true, y_pred, zero_division=0)

    if val_y_true is not None:
        opt_thr = compute_optimal_thr(val_f_pred, val_y_true)
        y_logits_pred = f_pred - opt_thr
        y_pred = np.where(f_pred > opt_thr, 1, 0)
        metrics[f'inst/bce_loss_opt'] = torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(y_logits_pred).float(), torch.from_numpy(y_true).float()).item()
        metrics[f'inst/acc_opt'] = accuracy_score(y_true, y_pred)
        metrics[f'inst/prec_opt'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'inst/rec_opt'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'inst/f1_opt'] = f1_score(y_true, y_pred, zero_division=0)

    metrics = {f'test/{k}' : v for k, v in metrics.items()}

    return metrics