"""PrimalConv: attention-based primal-graph convolution using dual-graph features.

Rewritten to extend MessagePassing directly (instead of GATConv) for
compatibility with PyTorch Geometric >= 2.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax


class PrimalConv(MessagePassing):
    r"""Modified graph-attention convolution on the primal graph whose
    attention coefficients are computed from the **dual**-graph node features.

    Args:
        in_channels (int): Input feature size per primal node.
        out_channels (int): Output feature size per primal node.
        out_channels_dual (int): Output feature size of the dual nodes
            (used to compute primal attention weights).
        concat_dual (bool): Whether dual heads are concatenated (True) or
            averaged (False).
        single_dual_nodes (bool): GraphCreator's ``single_dual_nodes``.
        undirected_dual_edges (bool): GraphCreator's ``undirected_dual_edges``.
        concat (bool): Concatenate (True) or average (False) primal heads.
        heads (int): Number of attention heads.
        negative_slope (float): LeakyReLU negative slope.
        dropout (float): Dropout on attention coefficients.
        bias (bool): Learn additive output bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_channels_dual: int,
        concat_dual: bool,
        single_dual_nodes: bool,
        undirected_dual_edges: bool,
        concat: bool = True,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ) -> None:
        if not concat_dual and heads != 1:
            raise ValueError(
                "Multiple primal heads require dual heads to be concatenated."
            )
        if single_dual_nodes:
            assert undirected_dual_edges, (
                "Single dual nodes require undirected dual edges."
            )
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self._single_dual_nodes = single_dual_nodes
        self._undirected_dual_edges = undirected_dual_edges
        self._out_channels_dual = out_channels_dual

        # Linear projection: (N, in_channels) → (N, heads * out_channels)
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameter: applied to dual features
        self.att = Parameter(torch.empty(1, heads, out_channels_dual))

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Storage for attention coefficients and dual features (set in forward)
        self._dual_features = None
        self._primal_edge_to_dual_node_idx = None
        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        glorot(self.att)
        zeros(self.bias)

    def reset_attention_parameters(self) -> None:
        glorot(self.att)

    def forward(
        self,
        x_primal: torch.Tensor,
        x_dual: torch.Tensor,
        edge_index_primal: torch.Tensor,
        primal_edge_to_dual_node_idx: dict,
        size=None,
    ):
        """Forward pass.

        Args:
            x_primal: ``(N_primal, in_channels)``
            x_dual: ``(N_dual, out_channels_dual)``
            edge_index_primal: ``(2, E_primal)``
            primal_edge_to_dual_node_idx: mapping from primal edge tuple to dual node index.
            size: optional (num_src, num_dst)

        Returns:
            out_primal: ``(N_primal, heads * out_channels)`` or ``(N_primal, out_channels)``
            primal_attention_coefficients: ``(E_primal, heads)``
        """
        assert isinstance(primal_edge_to_dual_node_idx, dict)

        # Project primal features
        x_primal = self.lin(x_primal)  # (N, heads * out_channels)

        # Store dual info for use in message()
        self._dual_features = x_dual
        self._primal_edge_to_dual_node_idx = primal_edge_to_dual_node_idx
        self._alpha = None

        out = self.propagate(edge_index_primal, size=size, x=x_primal)

        if self.bias is not None:
            out = out + self.bias

        return out, self._alpha

    def message(self, x_j, edge_index_i, edge_index_j, size_i):
        # edge_index_i: destination node indices (= row 1 of edge_index for source_to_target)
        # edge_index_j: source node indices
        i_list = edge_index_i.tolist()
        j_list = edge_index_j.tolist()

        if self._single_dual_nodes:
            # Config A: dual node {i, j}
            dual_indices = [
                self._primal_edge_to_dual_node_idx[tuple(sorted([ei, ej]))]
                for ei, ej in zip(i_list, j_list)
            ]
        else:
            # Configs B/C: dual node j→i
            dual_indices = [
                self._primal_edge_to_dual_node_idx[(ej, ei)]
                for ei, ej in zip(i_list, j_list)
            ]

        x_dual = self._dual_features[dual_indices].view(
            -1, self.heads, self._out_channels_dual
        )
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_dual * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        self._alpha = alpha.view(-1, self.heads)
        return (x_j * alpha.unsqueeze(-1)).view(-1, self.heads * self.out_channels)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads, self.out_channels)
        if self.concat:
            return aggr_out.view(-1, self.heads * self.out_channels)
        return aggr_out.mean(dim=1)

    def __repr__(self) -> str:
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
