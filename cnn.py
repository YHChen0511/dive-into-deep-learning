import torch.nn as nn
import torch
import torch.nn.functional as F

class CovLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 device=None,
                 dtype=None):

        super(CovLayer, self).__init__()

        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(
            padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(
            dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.weight = torch.rand(
            out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        if bias:
            self.bias = torch.rand(out_channels)
        else:
            self.bias = None

    def forward(self, x):
        batch_size, _, height, width = x.shape

        output_height = (height + 2 * self.padding[0] - self.dilation[0] * (
            self.kernel_size[0]-1)+1)/self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - self.dilation[1] * (
            self.kernel_size[1]-1)+1)/self.stride[1] + 1

        out = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=x.dtype)

        x_padded = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.out_channels):
                    out[:, k, i, j] = torch.sum(x_slice * self.weight[k, :, :, :], dim=(1, 2, 3))
                
                if self.bias is not None:
                    out[:, :, i, j] += self.bias
        
        return out
