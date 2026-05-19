import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import remove_self_loops


class GATConvNoSelfLoops(GATConv):
    r"""GATConv variant that does NOT add self-loops to the graph.

    Compatible with PyTorch Geometric >= 2.x.  The only difference from the
    standard :class:`GATConv` is that we remove any self-loops that may exist
    in ``edge_index`` and do NOT add new ones, instead of the default
    behaviour of always adding self-loops.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of attention heads. (default: 1)
        concat (bool, optional): Concatenate instead of average heads.
            (default: True)
        negative_slope (float, optional): LeakyReLU slope. (default: 0.2)
        dropout (float, optional): Attention coefficient dropout. (default: 0)
        bias (bool, optional): Learn additive bias. (default: True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ) -> None:
        # add_self_loops=False prevents GATConv from injecting self-loops
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            bias=bias,
            add_self_loops=False,
            **kwargs,
        )

    def forward(self, x, edge_index, size=None):
        # Remove any pre-existing self-loops before the attention computation
        edge_index, _ = remove_self_loops(edge_index)
        # Delegate to the modern GATConv forward (no self-loops will be added
        # because we passed add_self_loops=False in __init__)
        return super().forward(x, edge_index, size=size)

    def __repr__(self) -> str:
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )
