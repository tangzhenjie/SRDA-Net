import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy2d(input, target, weight_element, size_average=True):

    """

    :param input:  N C H W
    :param target: N 1 H W
    :param weight_element:  size(N * H * W)
    :param weight:
    :param size_average:
    :return:
    """
    n, c, h, w = input.size()
    nt, ct, ht, wt = target.size()

    # Handle inconsistent size between input and target
    """
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")
    """
    if h != ht or w != wt:
        raise Exception(
            "loss: wrong size between heatmap(%d,%d) and label(%d,%d)" %
            (h, w, ht, wt))
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    target = target.long()
    loss = F.nll_loss(
        log_p, target, ignore_index=250, weight=None, size_average=False, reduce=False)

    loss = loss * weight_element * 2
    if size_average:
        loss = loss.data.sum().float() / mask.data.sum().float()
    return loss


def bootstrapped_cross_entropy2d(input,
                                 target,
                                 K,
                                 weight=None,
                                 size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(
            log_p,
            target,
            weight=weight,
            ignore_index=250,
            reduce=False,
            size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average)
    return loss / float(batch_size)


def multi_scale_cross_entropy2d(input,
                                target,
                                weight=None,
                                size_average=True,
                                scale_weight=None):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        
        scale_weight = torch.pow(scale * torch.ones(n_inp),
                                 torch.arange(n_inp))
        scale_weight = scale_weight.cuda()
    

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average)
    
    return loss


if __name__ == "__main__":
    input = torch.randn(50, 5, 3, 3)
    target = torch.ones(50, 1, 3, 3)
    weight_element = torch.zeros(50 * 3 * 3)
    cross_entropy2d(input, target, weight_element)

