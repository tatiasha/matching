from math import log2
from torch import Tensor, sort


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == "const":
        return y_value
    if gain_scheme == "exp2":
        return 2**(y_value) - 1


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _, idx = sort(ys_pred, descending=True)
    return float(sum([float(compute_gain(i, gain_scheme))/log2(idx+2) for idx,i in enumerate(ys_true[idx])]))


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    dcg_val = dcg(ys_true, ys_pred, gain_scheme)
    dcg_ideal = dcg(ys_true, ys_true, gain_scheme)
    return dcg_val / dcg_ideal


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if ys_true.sum() == 0:
        return -1
    _, idx = sort(ys_pred, descending=True)
    return float(ys_true[idx][:k].sum() / min(k, ys_true.sum()))


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, idx = sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[idx]
    return 1 / (float(ys_true_sorted.argmax()) + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    _, idx = sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[idx]
    plook = 1
    p_rel_prev = ys_true_sorted[0]
    p_found_val = float(plook*p_rel_prev)
    for p_rel in ys_true_sorted[1:]:
        v = float(plook * (1 - p_rel_prev) * (1 - p_break) * p_rel)
        p_found_val += v
        plook = plook * (1 - p_rel_prev) * (1 - p_break)
        p_rel_prev = p_rel
    return p_found_val
    

def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    ys_pred_s, idx = sort(ys_pred, descending=True)
    ys_true_s = ys_true[idx]
    c = 0
    for i in range(0, len(ys_true)):
        for j in range(i+1, len(ys_true)):
            if (ys_pred_s[i] >= ys_pred_s[j]) != (ys_true_s[i] >= ys_true_s[j]):
                c += 1
    return c


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if ys_true.sum() == 0:
        return -1
    _, idx = sort(ys_pred, descending=True)
    s_ys_true = ys_true[idx]
    return float(sum([s_ys_true[i]*s_ys_true[:(i+1)].sum()/(i+1) for i in range(len(s_ys_true))])/s_ys_true.sum())