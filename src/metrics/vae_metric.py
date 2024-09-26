import torch

def batch_accuracy(pred, target):
    dim = pred.dim() - 1
    pred = pred.permute(0,2,1)
    target = target.permute(0,2,1)
    
    # 找到pred和target中第一个padding的位置
    pred_padding_pos = (pred[:, :, 4] > pred[:, :, :4].max(dim=-1)[0]).long().argmax(dim=1)
    target_padding_pos = (target[:, :, 4] == 1).long().argmax(dim=1)
    
    # 选择位置更大（靠后）的作为padding起点
    padding_start = torch.max(pred_padding_pos, target_padding_pos)
    
    # 创建非padding的mask
    seq_length = pred.size(1)
    mask = torch.arange(seq_length).unsqueeze(0).to(pred.device) < padding_start.unsqueeze(1)
    
    # 只比较非padding位置
    pred_argmax = torch.argmax(pred, dim=dim)[mask]
    target_argmax = torch.argmax(target, dim=dim)[mask]
    
    diff = pred_argmax - target_argmax
    
    return (len(diff) - torch.count_nonzero(diff)) / len(diff)

def batch_accuracy_bk(recons, batch_in):
    dim = recons.dim()-1
    batch_in = batch_in.permute(0,2,1)
    recons = recons.permute(0,2,1)
    
    # 找到每个序列中最后一个非padding位置
    non_padding_mask = (batch_in[:,:,4] != 1).float()  # 非padding位置为1，padding位置为0
    seq_lengths = non_padding_mask.sum(dim=1).long()
    
    # 创建一个mask，只包含非padding位置
    mask = torch.arange(batch_in.size(1)).unsqueeze(0).to(batch_in.device) < seq_lengths.unsqueeze(1)
    
    # 只比较非padding位置
    recons_argmax = torch.argmax(recons, dim=dim)[mask]
    batch_in_argmax = torch.argmax(batch_in, dim=dim)[mask]
    
    diff = recons_argmax - batch_in_argmax
    
    return (len(diff) - torch.count_nonzero(diff)) / len(diff)


if __name__ == "__main__":
    # 创建一个简单的测试用例
    recons = torch.tensor([
        [[0.1, 0.2, 0.6, 0.1, 0.0], [0.2, 0.5, 0.2, 0.1, 0.0], [0.1, 0.1, 0.1, 0.1, 0.7], [0.2, 0.2, 0.2, 0.2, 0.2]],
        [[0.7, 0.1, 0.1, 0.1, 0.0], [0.1, 0.6, 0.2, 0.1, 0.0], [0.2, 0.2, 0.2, 0.2, 0.4], [0.2, 0.2, 0.2, 0.2, 0.2]]
    ])
    
    batch_in = torch.tensor([
        [[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]
    ])

    print(recons.shape)
    accuracy = batch_accuracy(recons, batch_in)
    print(f"测试批次准确率: {accuracy:.4f}")