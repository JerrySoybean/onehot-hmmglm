import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def pad_for_convolve(spikes: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Pad tensor for convolution.

    Parameters
    ----------
    spikes : torch.Tensor of shape (..., n_time_bins, n_neurons)
        Input tensor.
    kernel_size : int
        Kernel size.

    Returns
    -------
    torch.Tensor of shape (..., n_time_bins + kernel_size - 1, n_neurons)
        Padded tensor.
    """

    return F.pad(spikes, (0, 0, kernel_size, -1), "constant", 0)


def pre_convolve(spikes: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Pre-convolve tensor.

    Parameters
    ----------
    spikes : torch.Tensor of shape (batch_size, n_time_bins, n_neurons)
        Input tensor.
    kernel : torch.Tensor of shape (kernel_size,)
        Kernel.

    Returns
    -------
    torch.Tensor
        Pre-convolved tensor.
    """

    kernel_size = kernel.size(-1)
    paded_spikes = pad_for_convolve(
        spikes, kernel_size
    )  # (batch_size, n_time_bins + kernel_size - 1, n_neurons)
    convolved_spikes = (
        F.conv1d(
            paded_spikes.permute(0, 2, 1).reshape(
                -1,
                1,
                paded_spikes.size(-2),
            ),  # (batch_size * n_neurons, 1, n_time_bins + kernel_size - 1)
            kernel[None, None, :],
        )  # (batch_size * n_neurons, 1, n_time_bins)
        .reshape(spikes.shape[0], spikes.shape[2], spikes.shape[1])
        .permute(0, 2, 1)
    )  # (batch_size, n_time_bins, n_neurons)

    return convolved_spikes


def optimal_permutation(tensor1, tensor2, num_classes=4):
    """
    计算 tensor2 的最佳类别置换，使其尽可能匹配 tensor1
    """
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()

    # 计算混淆矩阵（confusion matrix）
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = torch.sum((tensor1 == i) & (tensor2 == j))

    # 使用匈牙利算法找到最佳类别映射
    row_ind, col_ind = linear_sum_assignment(conf_matrix.numpy(), maximize=True)

    # 创建映射字典 {原类别: 置换后类别}
    mapping = {old: new for old, new in zip(col_ind, row_ind)}

    # 重新映射 tensor2
    tensor2_remapped = torch.tensor([mapping[x.item()] for x in tensor2]).reshape(
        tensor2.shape
    )

    return tensor2_remapped, mapping
