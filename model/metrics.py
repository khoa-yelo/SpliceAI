
from sklearn.metrics import average_precision_score

def topk_accuracy(probs, targets_onehot):
    topk_accs = []
    for c in [0, 1, 2]:
        probs_c = probs[:, c]
        targets_c = (targets_onehot[:, c]).astype(int)
        k = targets_c.sum()
        if k == 0:
            return float('nan')
        topk_indices = probs_c.argsort()[::-1][:k]
        correct = targets_c[topk_indices].sum()
        topk_acc = correct / k
        topk_accs.append(topk_acc)
    return topk_accs

def pr_auc(probs, targets_onehot):
    aucs = []
    for c in [0, 1, 2]:
        auc = average_precision_score(targets_onehot[:, c], probs[:, c])
        aucs.append(auc)
    return aucs