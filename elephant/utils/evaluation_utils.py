import numpy as np


def get_f1_score(confusion_matrix, average="micro", beta=1.0):

    tp_sum = np.diag(confusion_matrix)
    pred_sum = np.sum(confusion_matrix, axis=0)
    true_sum = np.sum(confusion_matrix, axis=1)
    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])
    beta2 = beta ** 2
    # Divide, and on zero-division, set scores and/or warn according to zero_division:
    precision = _prf_divide(tp_sum, pred_sum)
    recall = _prf_divide(tp_sum, true_sum)

    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall
        denom[denom == 0.0] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    precision = np.average(precision)
    recall = np.average(recall)
    f_score = np.average(f_score)
    return precision, recall, f_score


def _prf_divide(
    numerator, denominator, zero_division="warn"
):
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ["warn", 0] else 1.0
    return result
