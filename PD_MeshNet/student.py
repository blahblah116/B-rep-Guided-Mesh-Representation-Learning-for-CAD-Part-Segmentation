"""PD-MeshNet student model compatible with the B2Mesh distillation framework."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .models import DualPrimalMeshSegmenter


class PDMeshNetStudent(nn.Module):
    """Primal-Dual MeshNet encoder + embedding head + classifier.

    Wraps :class:`DualPrimalMeshSegmenter` so that it has the same
    ``forward`` signature expected by B2Mesh's training loop::

        logits, face_embeddings = student(sample)

    where ``sample`` is the dict produced by
    :class:`MFCADPlusPlusDualPrimal` or :class:`Fusion360DualPrimal`.

    Architecture
    ------------
    * A stack of ``DualPrimalDownConv`` blocks (encoder) whose last output
      has ``embedding_dim`` primal channels.
    * A linear **embedding head** that maps primal-node features to a
      fixed-size embedding (identity if the encoder already outputs
      ``embedding_dim``).
    * A linear **classification head** that maps embeddings to class logits.

    The encoder is built by setting ``conv_primal_out_res[-1] = embedding_dim``
    inside ``DualPrimalMeshSegmenter`` and using ``do_not_add_final_block=True``.
    A separate classifier is then applied on top.

    Args:
        in_channels_primal (int): Feature dimension of each primal node
            (mesh face) from the GraphCreator output.
        in_channels_dual (int): Feature dimension of each dual node.
        num_classes (int): Number of segmentation classes.
        embedding_dim (int): Dimension of the per-face distillation embedding.
        conv_primal_out_res (list[int]): Output channels of each encoder block's
            primal convolution. The last entry is replaced by ``embedding_dim``.
        conv_dual_out_res (list[int]): Output channels of each encoder block's
            dual convolution.
        single_dual_nodes (bool): Must match the dataset GraphCreator option.
        undirected_dual_edges (bool): Must match the dataset GraphCreator option.
        fractions_primal_edges_to_keep (list[float | None] | None):
            Fraction of primal edges to keep per pooling block. ``None`` means
            no pooling is applied.
        heads (int): GAT attention heads per conv layer.
        use_res_blocks (bool): Use residual down-conv blocks.
    """

    def __init__(
        self,
        in_channels_primal: int,
        in_channels_dual: int,
        num_classes: int,
        embedding_dim: int = 128,
        conv_primal_out_res: list[int] | None = None,
        conv_dual_out_res: list[int] | None = None,
        single_dual_nodes: bool = True,
        undirected_dual_edges: bool = True,
        fractions_primal_edges_to_keep: list[float | None] | None = None,
        heads: int = 1,
        use_res_blocks: bool = False,
    ) -> None:
        super().__init__()

        if conv_primal_out_res is None:
            conv_primal_out_res = [embedding_dim, embedding_dim]
        if conv_dual_out_res is None:
            conv_dual_out_res = [embedding_dim, embedding_dim]

        # Force the last primal output to embedding_dim so we can attach our head
        conv_primal_out_res = list(conv_primal_out_res[:-1]) + [embedding_dim]

        if fractions_primal_edges_to_keep is None:
            fractions_primal_edges_to_keep = [None] * len(conv_primal_out_res)

        self.encoder = DualPrimalMeshSegmenter(
            in_channels_primal=in_channels_primal,
            in_channels_dual=in_channels_dual,
            conv_primal_out_res=conv_primal_out_res,
            conv_dual_out_res=conv_dual_out_res,
            num_classes=num_classes,  # unused when do_not_add_final_block=True
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            use_dual_primal_res_down_conv_blocks=use_res_blocks,
            fractions_primal_edges_to_keep=fractions_primal_edges_to_keep,
            heads=heads,
            do_not_add_final_block=True,
            log_ratios_new_old_primal_nodes=True,
        )

        # Determine actual embedding size (heads * embedding_dim if concat=True)
        actual_emb_dim = embedding_dim * heads
        self.embedding_proj = (
            nn.Identity()
            if actual_emb_dim == embedding_dim
            else nn.Linear(actual_emb_dim, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, sample: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            sample: Dict with keys ``primal_graph``, ``dual_graph``, ``petdni``.
                As returned by the dataset ``__getitem__`` (single sample or
                already collated into a batch via ``collate_pd_samples``).

        Returns:
            logits: ``(N_faces, num_classes)``
            embeddings: ``(N_faces, embedding_dim)``
        """
        primal_graph = sample["primal_graph"]
        dual_graph = sample["dual_graph"]
        petdni = sample["petdni"]

        (
            primal_graph_out,
            _dual_graph_out,
            _petdni_out,
            node_to_cluster,
            _ratios_nodes,
            _ratios_edges,
            _primal_before_pool,
            _dual_before_pool,
            _pool_info,
        ) = self.encoder.forward_encoder_only(
            primal_graph_batch=primal_graph,
            dual_graph_batch=dual_graph,
            primal_edge_to_dual_node_idx_batch=petdni,
        )

        # Unpool back to original face resolution
        if len(node_to_cluster) == 0:
            face_features = primal_graph_out.x
        else:
            keys = sorted(node_to_cluster.keys())[::-1]
            mapping = node_to_cluster[keys[0]]
            for k in keys[1:]:
                mapping = mapping[node_to_cluster[k]]
            face_features = primal_graph_out.x[mapping]

        embeddings = self.embedding_proj(face_features)
        logits = self.classifier(embeddings)
        return logits, embeddings


# ---------------------------------------------------------------------------
# Collation helpers for the B2Mesh DataLoader
# ---------------------------------------------------------------------------

def collate_pd_samples(batch: list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    """Collate a list of PD-MeshNet samples.

    PD-MeshNet primal/dual graphs cannot be naively stacked because they have
    variable topology (edges, node counts).  We keep each sample separate and
    return a list; the training loop iterates over it.  When batch_size=1
    (the recommended setting) a plain dict is returned.
    """
    # Batch each B-rep graph individually so the teacher can forward on it
    out = []
    for s in batch:
        s = dict(s)
        if "brep_graph" in s:
            import dgl
            s["brep_graph"] = dgl.batch([s["brep_graph"]])
        out.append(s)
    return out[0] if len(out) == 1 else out
