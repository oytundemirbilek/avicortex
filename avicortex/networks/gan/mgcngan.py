"""Define MGCN-GAN architecture."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    LeakyReLU,
    Linear,
    Module,
)

# from torch.modules.module import Module
from torch.nn.parameter import Parameter


class BatchNormLinear(Module):
    """"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()

        self.batchnorm_layer = BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return self.batchnorm_layer(x)


class BatchNorm(BatchNormLinear):
    """"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class BatchNormAdj(BatchNormLinear):
    """"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        batch_size = x.size(0)
        num_region = x.size(1)
        x = x.contiguous().view(batch_size, -1)
        x = super().forward(x)
        x = x.contiguous().view(batch_size, num_region, -1)
        return x


class LayerNorm(Module):
    """"""

    def __init__(
        self, num_features: int, eps: float = 1e-05, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        # num_features = (input.size()[1:])
        self.layer_norm = LayerNorm(num_features, eps, elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return self.layer_norm(x)


class GraphConvolution(Module):
    """"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """"""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """"""
        support = torch.matmul(x, self.weight)
        output = torch.einsum("bij,bjd->bid", [adj, support])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self) -> str:
        """"""
        return (
            f"{self.__class__.__name__} ({self.in_features}) -> ({self.out_features})"
        )


class GCNGenerator(Module):
    def __init__(self, in_feature: int):
        super().__init__()

        self.gc1 = GraphConvolution(in_feature, in_feature)
        self.LayerNorm1 = LayerNorm([in_feature, in_feature])

        self.gc2_01 = GraphConvolution(in_feature, int(in_feature * 2))
        self.LayerNorm2_01 = LayerNorm([in_feature, int(in_feature * 2)])
        self.gc2_12 = GraphConvolution(int(in_feature * 2), in_feature)
        self.LayerNorm2_12 = LayerNorm([in_feature, in_feature])

        self.gc3_01 = GraphConvolution(in_feature, int(in_feature / 2))
        self.LayerNorm3_01 = LayerNorm([in_feature, int(in_feature / 2)])
        self.gc3_13 = GraphConvolution(int(in_feature / 2), in_feature)
        self.LayerNorm3_13 = LayerNorm([in_feature, in_feature])

        # the theta: can compare different initializations
        self.weight = Parameter(
            torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        )

    def forward(self, topo: Tensor, funcs: Tensor, test_mode: bool = False) -> Tensor:
        """"""
        topo = funcs  # to compare with different updating methods: topo != funcs

        x1 = self.gc1(funcs, topo)
        x1 = self.LayerNorm1(x1)
        x1 = F.leaky_relu(x1, 0.05, inplace=True)

        x2 = self.gc2_01(funcs, topo)
        x2 = self.LayerNorm2_01(x2)
        x2 = F.leaky_relu(x2, 0.05, inplace=True)
        x2 = self.gc2_12(x2, topo)
        x2 = self.LayerNorm2_12(x2)
        x2 = F.leaky_relu(x2, 0.05, inplace=True)

        x3 = self.gc3_01(funcs, topo)
        x3 = self.LayerNorm3_01(x3)
        x3 = F.leaky_relu(x3, 0.05, inplace=True)
        x3 = self.gc3_13(x3, topo)
        x3 = self.LayerNorm3_13(x3)
        x3 = F.leaky_relu(x3, 0.05, inplace=True)

        x = self.weight[0] * x1 + self.weight[1] * x2 + self.weight[2] * x3
        outputs = x + torch.transpose(x, 1, 2)
        if test_mode:
            return outputs.squeeze().unsqueeze(0)
        return outputs.squeeze()


class Discriminator(Module):
    """"""

    def __init__(
        self,
        in_feature: int,
        out1_feature: int,
        out2_feature: int,
        out3_feature: int,
        dropout: float,
    ):
        super().__init__()

        self.gc1 = GraphConvolution(in_feature, out1_feature)
        self.batchnorm1 = BatchNorm(out1_feature)
        self.gc2 = GraphConvolution(out1_feature, out2_feature)
        self.batchnorm2 = BatchNorm(out2_feature)
        self.gc3 = GraphConvolution(out2_feature, out3_feature)
        self.batchnorm3 = BatchNorm(out3_feature)
        self.batchnorm4 = BatchNormLinear(1024)
        self.dropout = dropout
        self.Linear1 = Linear(
            out3_feature * in_feature, 1024
        )  # 148 for atlas1 and 68 for atlas2
        self.dropout = dropout
        self.Linear2 = Linear(1024, 1)

    @staticmethod
    def batch_eye(size: int | tuple[int, int]) -> Tensor:
        """"""
        if isinstance(size, int):
            size = size, size
        batch_size, n = size
        identity = torch.eye(n).unsqueeze(0)
        return identity.repeat(batch_size, 1, 1)

    def forward(self, adj_matrix: Tensor, batch_size: int) -> Tensor:
        """"""
        x = self.batch_eye(adj_matrix.shape).to(adj_matrix.device).float()

        x = self.gc1(x, adj_matrix)
        x = LeakyReLU(0.2, True)(x)
        x = self.batchnorm1(x)

        x = self.gc2(x, adj_matrix)
        x = LeakyReLU(0.2, True)(x)
        x = self.batchnorm2(x)

        x = self.gc3(x, adj_matrix)
        x = LeakyReLU(0.2, True)(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.contiguous().view(batch_size, -1)
        x = self.Linear1(x)
        x = LeakyReLU(0.2, True)(x)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.Linear2(x)
        x = torch.sigmoid(x)
        x = x.contiguous().view(batch_size, -1)
        outputs = x
        return outputs.squeeze()
