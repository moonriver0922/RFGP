# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
"""RFGP model: MAE+MoE spectrum encoder with per-scene NeRF2 rendering."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from timm.models.vision_transformer import Block, PatchEmbed

import renderer
from nerf2 import NeRF2
from paths import merge_renderer_cfg
from pos_embed import get_2d_sincos_pos_embed


class Expert(nn.Module):
    """Single feed-forward expert used inside MoE."""

    def __init__(self, hidden_dim, expert_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_dim)
        self.fc2 = nn.Linear(expert_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MoE(nn.Module):
    """Sparse Mixture-of-Experts with a shared expert and load-balancing loss.

    Parameters
    ----------
    hidden_dim : int
        Token feature dimension.
    expert_dim : int
        Hidden width inside each expert MLP.
    num_experts : int
        Number of routed experts.
    top_k : int
        Number of experts activated per token.
    """

    def __init__(self, hidden_dim, expert_dim, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(hidden_dim, expert_dim) for _ in range(num_experts)])
        self.shared_expert = Expert(hidden_dim, expert_dim)
        self.gating = nn.Linear(hidden_dim, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        """Route tokens to top-k experts and return MoE output + aux loss.

        Parameters
        ----------
        x : torch.Tensor
            [batch, seq_len, hidden_dim]

        Returns
        -------
        output : torch.Tensor
            [batch, seq_len, hidden_dim]
        auxiliary_loss : torch.Tensor
            Scalar load-balancing loss.
        expert_activations : torch.Tensor
            [num_experts] activation counts.
        """
        # Gating weights over experts
        logits = self.gating(x)  # [B, L, E]
        weights = F.softmax(logits, dim=-1)

        # Select top-k experts and renormalize
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)

        # Weighted sum of selected expert outputs
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_weights[:, :, i]
            expert_output = torch.stack(
                [self.experts[j](x) for j in range(self.num_experts)], dim=-1)
            expert_output = torch.gather(
                expert_output, -1,
                expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(-1), -1))
            expert_output = expert_output.squeeze(-1)
            output += expert_weight.unsqueeze(-1) * expert_output

        output += self.shared_expert(x)
        auxiliary_loss = self.compute_auxiliary_loss(weights, top_k_indices)
        expert_activations = self.compute_expert_activations(top_k_indices)
        return output, auxiliary_loss, expert_activations

    def compute_auxiliary_loss(self, weights, top_k_indices):
        """Switch-style load balancing: encourage uniform expert usage.

        Parameters
        ----------
        weights : torch.Tensor
            Softmax gate probabilities [B, L, E].
        top_k_indices : torch.Tensor
            Selected expert indices [B, L, K].

        Returns
        -------
        torch.Tensor
            Scalar auxiliary loss.
        """
        batch_size, seq_len, _ = weights.shape
        num_experts = self.num_experts

        # c_i: number of tokens assigned to expert i
        expert_counts = torch.zeros(num_experts, device=weights.device)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            src = torch.ones_like(expert_idx.flatten()).contiguous()
            expert_counts = expert_counts.to(torch.float32)
            src = src.to(torch.float32)
            expert_counts.scatter_add_(0, expert_idx.flatten(), src)

        # m_i: sum of gate probabilities for expert i
        m_i = weights.sum(dim=0).sum(dim=0)
        L_aux = (num_experts / ((batch_size * seq_len) ** 2)) * torch.sum(expert_counts * m_i)
        return L_aux

    def compute_expert_activations(self, top_k_indices):
        """Count how many tokens activated each expert.

        Parameters
        ----------
        top_k_indices : torch.Tensor
            [batch, seq_len, top_k] selected expert indices.

        Returns
        -------
        torch.Tensor
            [num_experts] activation counts.
        """
        expert_activations = torch.zeros(self.num_experts, device=top_k_indices.device)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            src = torch.ones_like(expert_idx.flatten())
            expert_activations = expert_activations.to(torch.float32)
            src = src.to(torch.float32)
            expert_activations.scatter_add_(0, expert_idx.flatten(), src)
        return expert_activations


class TransformerBlockWithMoE(nn.Module):
    """Transformer block using multi-head attention + MoE FFN."""

    def __init__(self, hidden_dim=1024, num_heads=8, expert_dim=256 * 4 * 4,
                 num_experts=15, top_k=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.moe = MoE(hidden_dim, expert_dim, num_experts, top_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            [batch, seq_len, hidden_dim]

        Returns
        -------
        x : torch.Tensor
            Updated tokens.
        loss : torch.Tensor
            MoE auxiliary loss.
        count : torch.Tensor
            Expert activation counts.
        """
        attn_output, _attn_weight = self.attention(x, x, x, average_attn_weights=False)
        x = x + attn_output
        x = self.norm1(x)
        moe_output, loss, count = self.moe(x)
        x = x + moe_output
        x = self.norm2(x)
        return x, loss, count


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with ViT backbone; every 4th block uses MoE.

    Spectrum maps are treated as single-channel images of shape (9, 36).
    The forward return uses the mean encoder latent (excluding CLS) as the
    position embedding for downstream NeRF rendering.
    """

    def __init__(self, img_size=(9, 36), patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            if (i + 1) % 4 != 0 else
            TransformerBlockWithMoE(hidden_dim=embed_dim, num_heads=num_heads)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize fixed sin-cos pos embeds and linear layers."""
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
            cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert images to patch tokens.

        Parameters
        ----------
        imgs : torch.Tensor
            (N, C, H, W)

        Returns
        -------
        torch.Tensor
            (N, L, patch_size**2 * C)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, h, w):
        """Convert patch tokens back to image tensors.

        Parameters
        ----------
        x : torch.Tensor
            (N, L, patch_size**2 * C)
        h, w : int
            Spatial height/width before patchification.
        """
        p = self.patch_embed.patch_size[0]
        h = h // p
        w = w // p
        c = x.shape[2] // p ** 2
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """Per-sample random masking via noise argsort.

        Parameters
        ----------
        x : torch.Tensor
            [N, L, D] patch tokens.
        mask_ratio : float
            Fraction of patches to mask.

        Returns
        -------
        x_masked, mask, ids_restore
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """Encode masked spectrum patches; accumulate MoE aux loss."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        loss = 0
        count = None
        for i, blk in enumerate(self.blocks):
            if (i + 1) % 4 == 0:
                x, load_balancing_loss, count_expert = blk(x)
                loss += load_balancing_loss
                count = count_expert if count is None else count + count_expert
            else:
                x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore, loss, count

    def forward_decoder(self, x, ids_restore):
        """Decode latent tokens to patch predictions (for visualization)."""
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """MAE reconstruction loss on masked patches only."""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """Encode a flattened spectrum map and return latent + mask viz.

        Parameters
        ----------
        imgs : torch.Tensor
            [N, 9*36] or compatible flattened spectra.
        mask_ratio : float
            MAE mask ratio.

        Returns
        -------
        pred : torch.Tensor
            Mean latent (excl. CLS), used as position feature input.
        mask : torch.Tensor
            Binary mask over patches.
        masked_imgs : torch.Tensor
            Spectrum with masked patches zeroed (for visualization).
        loss : torch.Tensor
            MoE auxiliary loss.
        count : torch.Tensor
            Expert activation counts.
        """
        H = 36
        W = 9
        imgs = imgs.reshape(shape=(imgs.shape[0], 1, W, H))
        latent, mask, ids_restore, loss, count = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        pred = self.unpatchify(pred, W, H)
        imgs = self.patchify(imgs)
        masked_imgs = imgs * (1 - mask).unsqueeze(-1)
        masked_imgs = self.unpatchify(masked_imgs, W, H)
        pred = latent[:, 1:, :].mean(dim=1, keepdim=True)
        return pred, mask, masked_imgs, loss, count


class MLP(nn.Module):
    """Two-layer MLP with ReLU."""

    def __init__(self, input_dim=1024, inner_dim=256, output_dim=16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, output_dim),
        )
        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class PREDICTION_HEAD(nn.Module):
    """MLP head for coordinate regression (downstream / unused in pretrain)."""

    def __init__(self, input_dim=32, inner_dim=8, output_dim=3):
        super().__init__()
        self.net = MLP(input_dim, inner_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class NeRF2_HEAD(nn.Module):
    """Project MAE latents to NeRF2 transmitter features."""

    def __init__(self, input_dims=324, inner_dims=128, output_dims=32):
        super().__init__()
        self.net = MLP(input_dims, inner_dims, output_dims)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mean, logvar):
        """VAE-style reparameterization (available for variants)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            MAE latent features.

        Returns
        -------
        torch.Tensor
            Position / transmitter feature for rendering.
        """
        return self.net(x)


def gen_rays_spectrum(gateway_pos, gateway_orientation, a_split=36, e_split=9,
                      a_s=0, a_e=36, e_s=0, e_e=9):
    """Generate rays from a gateway covering the spectrum angular grid.

    Parameters
    ----------
    gateway_pos : torch.Tensor
        [B, 3] gateway positions in world coordinates.
    gateway_orientation : torch.Tensor
        [B, 4] gateway quaternion.
    a_split, e_split : int
        Azimuth / elevation resolution of the spectrum map.
    a_s, a_e, e_s, e_e : int
        Angular index ranges (defaults cover the full map).

    Returns
    -------
    r_o : torch.Tensor
        [n_rays, 3] ray origins.
    r_d : torch.Tensor
        [n_rays, 3] unit ray directions in world coordinates.
    """
    azimuth = torch.linspace(0, np.pi * 2 * (1 - 1 / a_split), a_split)
    elevation = torch.linspace(0, np.pi / 2 * (1 - 1 / e_split), e_split)
    azimuth = torch.tile(azimuth, (e_e - e_s,))
    elevation = torch.repeat_interleave(elevation, a_e - a_s)
    x = 1 * torch.cos(elevation) * torch.cos(azimuth)
    y = 1 * torch.cos(elevation) * torch.sin(azimuth)
    z = 1 * torch.sin(elevation)

    r_d = torch.stack([x, y, z], dim=0)
    R = torch.from_numpy(
        Rotation.from_quat(gateway_orientation.cpu().detach().numpy()).as_matrix()
    ).float()
    r_d = R @ r_d
    r_o = torch.tile(gateway_pos, ((a_e - a_s) * (e_e - e_s),)).reshape(-1, 3)
    r_d = r_d.permute(0, 2, 1).reshape(-1, 3)
    return r_o, r_d


class Decoder(nn.Module):
    """Stack of ViT blocks (optional context path unused in pretrain)."""

    def __init__(self, embed_dim=32, depth=8, num_heads=4, mlp_ratio=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.net = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, context=None):
        for blk in self.net:
            if context is None:
                x = blk(x)
            else:
                x = blk(x, context)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention used by optional cross-attention blocks."""

    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"
        self.values = nn.Linear(324, embed_size, bias=False)
        self.keys = nn.Linear(324, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        q = self.queries(q)
        k = self.keys(k)
        v = self.values(v)
        N = q.shape[0]
        value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]
        values = v.view(N, value_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = k.view(N, key_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = q.view(N, query_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        out = torch.einsum("nhqk,nhvd->nhqd", [attention, values])
        out = out.permute(0, 2, 1, 3).reshape(N, query_len, self.embed_size)
        return self.fc_out(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embed_size, ff_hidden_size, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class CrossAttentionTransformerBlock(nn.Module):
    """Cross-attention block (kept for downstream variants)."""

    def __init__(self, embed_size, heads, mlp_ratio, dropout=0.):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, embed_size * mlp_ratio, dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x, context, mask=None):
        attention = self.attention(x, context, context, mask)
        x = self.dropout1(attention) + x
        x = self.norm1(x)
        forward = self.feed_forward(x)
        x = self.dropout2(forward) + x
        x = self.norm2(x)
        return x


class RFGP(nn.Module):
    """RFGP: MAE+MoE encoder + per-scene NeRF2 spectrum renderer.

    One NeRF2 + renderer is created for each discovered ``scene_id``. Renderer
    hyperparameters come from ``renderer.defaults`` with optional
    ``renderer.overrides[scene_id]``.
    """

    def __init__(self, scene_ids, **kwargs):
        """
        Parameters
        ----------
        scene_ids : sequence of int
            Scene ids present in the training/eval data.
        **kwargs
            Must include ``mae`` and ``renderer`` config dicts.
        """
        super().__init__()
        if not scene_ids:
            raise ValueError("RFGP requires a non-empty scene_ids list")

        mae_kwargs = kwargs['mae']
        renderer_kwargs = kwargs['renderer']
        self.mae = MaskedAutoencoderViT(**mae_kwargs)
        self.scene_ids = [int(s) for s in scene_ids]

        self.nerfs = nn.ModuleDict({
            str(sid): NeRF2() for sid in self.scene_ids
        })
        self.proj = NeRF2_HEAD(input_dims=256 * 4, inner_dims=128 * 2, output_dims=32)

        render_cls = renderer.renderer_dict[renderer_kwargs.get('mode', 'spectrum')]
        # Plain dict: Renderer is not an nn.Module; gradients flow via nerfs.
        self.renderers = {
            str(sid): render_cls(
                networks_fn=self.nerfs[str(sid)],
                **merge_renderer_cfg(renderer_kwargs, sid),
            )
            for sid in self.scene_ids
        }

    def _scene_key(self, scenes) -> str:
        if torch.is_tensor(scenes):
            sid = int(scenes.reshape(-1)[0].item())
        else:
            sid = int(scenes)
        key = str(sid)
        if key not in self.renderers:
            raise KeyError(
                f"No NeRF/renderer registered for scene id {sid}. "
                f"Known scenes: {self.scene_ids}"
            )
        return key

    def forward(self, imgs, scenes, gateways, mode):
        """Encode spectra and optionally render predicted signal maps.

        Parameters
        ----------
        imgs : torch.Tensor
            [B, n_gateways, 9*36] input spectra.
        scenes : int or torch.Tensor
            Scene id selecting which NeRF/renderer to use.
        gateways : torch.Tensor
            [B, n_gateways, 7] xyz + quaternion per gateway.
        mode : int
            1 = render spectrum via NeRF; other values reserved.

        Returns
        -------
        When mode == 1:
            predict_signal, pos_f, masked_imgs, moe_loss, expert_counts
        """
        gateway_nums = gateways.shape[1]
        scene_renderer = self.renderers[self._scene_key(scenes)]
        spt_nums = imgs.shape[-1]
        latent, mask, masked_imgs, loss, count = self.mae(imgs)
        pos_f = self.proj(latent)
        if mode == 1:
            gateways = gateways.reshape(-1, gateway_nums, 7)
            predict_signals = []
            for i in range(gateway_nums):
                pos_feature = pos_f.repeat(1, spt_nums, 1).reshape(-1, pos_f.shape[-1])
                gateway_pos = gateways[:, i, :3].reshape(-1, 3)
                gateway_orientation = gateways[:, i, 3:7].reshape(-1, 4)
                r_o, r_d = gen_rays_spectrum(gateway_pos, gateway_orientation)
                r_o, r_d = r_o.to(pos_feature.device), r_d.to(pos_feature.device)
                predict_signal = scene_renderer.render_ss(
                    pos_feature, r_o, r_d).reshape(-1, 1, 9, 36)
                predict_signals.append(predict_signal)
            predict_signal = torch.cat(predict_signals, dim=0)
            return predict_signal, pos_f, masked_imgs, loss, count


# Backward-compatible alias
LocGPT2 = RFGP
