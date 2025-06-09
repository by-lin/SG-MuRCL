import numpy as np
import matplotlib.pyplot as plt

def plot_att_hist(f_pred, y_true, T_true, bag_idx):
    """
    Args:
        f_pred: (num_inst,) attention val of the instances
        y_true: (num_inst,) label of the instances
        T_true: (num_bags,) label of the bags
        bag_idx: (num_inst,) maps each instance to its bag    
    """
    
    pos_bags_idx = np.where(T_true == 1)[0] # positive bags

    idx_keep = np.isin(bag_idx, pos_bags_idx)

    y_true = y_true[idx_keep]
    f_pred = f_pred[idx_keep]
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    pos_inst = f_pred[pos_idx]
    neg_inst = f_pred[neg_idx]
    
    fig, ax = plt.subplots()
    counts_neg, bins_neg = np.histogram(neg_inst, bins=20, density=False)
    counts_neg = counts_neg / counts_neg.sum()
    ax.hist(bins_neg[:-1], bins_neg, weights=counts_neg, label='Negative instances', edgecolor='black', alpha=0.7, color='tab:green')

    counts_pos, bins_pos = np.histogram(pos_inst, bins=20, density=False)
    counts_pos = counts_pos / counts_pos.sum()
    ax.hist(bins_pos[:-1], bins_pos, weights=counts_pos, label='Positive instances', edgecolor='black', alpha=0.7, color='tab:red')

    ax.set_ylim(0, 1)

    ax.set_xlabel('Attention value')
    ax.set_ylabel('Frequency')
    ax.legend()

    return fig