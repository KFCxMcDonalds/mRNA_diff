import torch

def batch_accuracy(pred, target):
    pred = pred.view(-1, 5)
    target = target.view(-1, 5)
    diff = torch.argmax(pred, dim=-1) - torch.argmax(target, dim=-1)

    return (len(diff) - torch.count_nonzero(diff)) / len(diff)