import torch.nn.functional as F
from torch import nn

import mixmatch.models.utils as utils


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    block_repeats = (depth - 4) // 6

    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], block_repeats),
        'group1': gen_group_params(widths[0], widths[1], block_repeats),
        'group2': gen_group_params(widths[1], widths[2], block_repeats),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, stride):
        print(f"\t\tBN -> ReLU = X")
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0'), inplace=True)
        print(f"\t\tConv {params[base + '.conv0'].shape} Stride {stride} Pad 1")
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        print(f"\t\tBN -> ReLU")
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1'), inplace=True)
        print(f"\t\tConv {params[base + '.conv1'].shape} Stride 1 Pad 1 = Z")
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)

        if base + '.convdim' in params:
            print(f"\t\t\tX -> Conv {params[base + '.convdim'].shape} Stride {stride} Pad 0")
            print(f"\t\t\tZ + X")
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            print(f"\t\t\tZ + X")
            return z + x

    def group(o, params, base, stride):
        for i in range(block_repeats):
            print(f"\tBlock {i}")
            o = block(o, params, '%s.block%d' % (base, i),
                      stride if i == 0 else 1)
        return o

    def f(input, params):
        print(f"Conv {params['conv0'].shape} Stride 1 Pad 1")
        x = F.conv2d(input, params['conv0'], padding=1)
        print(f"Group 0")
        g0 = group(x, params, 'group0', 1)
        print(f"Group 1")
        g1 = group(g0, params, 'group1', 2)
        print(f"Group 2")
        g2 = group(g1, params, 'group2', 2)
        print(f"BN -> ReLU")
        o = F.relu(utils.batch_norm(g2, params, 'bn'))
        print(f"AvgPool 8 Stride 1 Pad 0")
        o = F.avg_pool2d(o, 8, 1, 0)
        print(f"View")
        o = o.view(o.size(0), -1)
        print(f"Linear {params['fc.weight'].shape}")
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f, flat_params


f, p = resnet(28, 2, 10)
import torch

a = f(torch.rand(16, 3, 32, 32), p, )


# 6x
# 1 Block:
# X --> BN --> ReLU
# --> Conv2D Stride ? Pad 1
# --> BN --> ReLU
# --> Conv2D Stride 1 Pad 1 --> Y
# If ConvDim : X + Y --> Conv2D Stride ? Pad 0
# Else       : X + Y
#
# The ConvDim is to match the dimension, when we change blacks
#
#
#
# class Block(nn.Module):
#     def __init__(
#             self,
#             dims: tuple[int, int, int] | tuple[int, int, int, int],
#             ksizes: tuple[int, int] | tuple[int, int, int] = (3, 3, 3),
#             strides: tuple[int, int] | tuple[int, int, int] = (2, 1, 2),
#             pads: tuple[int, int] | tuple[int, int, int] = (1, 1, 0),
#     ):
#         """
#
#         Args:
#             dims:
#             ksizes:
#             strides:
#             pads:
#         """
#         super().__init__()
#         assert len(dims) in (3, 4), ("Only supply 3 or 4 dimensions. "
#                                      "See docstring for more info.")
#         self.bn0 = nn.BatchNorm2d(dims[0])
#         self.relu0 = nn.ReLU()
#         self.conv0 = nn.Conv2d(
#             dims[0], dims[1], ksizes[0],
#             stride=strides[0], padding=pads[0]
#         )
#         self.bn1 = nn.BatchNorm2d(dims[1])
#         self.relu1 = nn.ReLU()
#         self.conv1 = nn.Conv2d(
#             dims[1], dims[2], ksizes[1],
#             stride=strides[1], padding=pads[1]
#         )
#         if len(dims) == 4:
#             self.conv_proj = nn.Conv2d(
#                 dims[2], dims[3], ksizes[2],
#                 stride=strides[2], padding=pads[2]
#             )
#         else:
#             self.conv_proj = None
#
#     def forward(self, x):
#         x0 = self.relu0(self.bn0(x))
#         x1 = self.conv0(x0)
#         x1 = self.conv1(self.relu1(self.bn1(x1)))
#         if self.conv_proj is not None:
#             x0 = self.conv_proj(x1)
#         return x + x_
#
#
# class Group(nn.Module):
#     def __init__(
#             self,
#             dim_in: int,
#             dim_block: int,
#             dim_out: int,
#             n_blocks: int = 6,
#             stride: int = 1,
#     ):
#         super().__init__()
#         self.blocks = nn.Sequential(
#             Block((dim_in, dim_block, dim_block), strides=(stride, stride)),
#             *[Block((dim_block, dim_block, dim_block),
#                     strides=(stride, stride)) for _ in range(n_blocks - 2)],
#             Block((dim_block, dim_block, dim_block, dim_out),
#                   strides=(stride, stride, stride)),
#         )
#
#     def forward(self, x):
#         return self.blocks(x)
#
# Group(16, 32, 64)(torch.rand(16, 16, 32, 32)).shape
