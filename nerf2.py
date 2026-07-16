# -*- coding: utf-8 -*-
"""NeRF2 network for wireless spectrum rendering.

Positional encoding and attenuation/signal MLPs used by RFGP.
Defaults match the RFGP pretraining configuration (tx feature dim 32).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Misc loss helpers (used by external scripts)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
img2me = lambda x, y: torch.mean(abs(x - y))
sig2mse = lambda x, y: torch.mean((x - y) ** 2)
csi2snr = lambda x, y: -10 * torch.log10(
    torch.norm(x - y, dim=(1, 2)) ** 2 /
    torch.norm(y, dim=(1, 2)) ** 2
)


class Embedder:
    """Fourier positional encoding (NeRF-style gamma)."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """Build sin/cos embedding functions from kwargs."""
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """Apply positional encoding to inputs.

        Returns
        -------
        torch.Tensor
            Concatenated embedded features along the last dimension.
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, is_embeded=True, input_dims=3):
    """Build a positional encoding function.

    Parameters
    ----------
    multires : int
        Number of frequency bands (L); max freq is 2^(L-1).
    is_embeded : bool
        If False, return identity and original input_dims.
    input_dims : int
        Dimensionality of the input coordinates/features.

    Returns
    -------
    embed : callable
        Embedding function.
    out_dim : int
        Output feature dimension after embedding.
    """
    if is_embeded is False:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class NeRF2(nn.Module):
    """NeRF2 attenuation + signal MLP for wireless rendering.

    Parameters
    ----------
    D : int
        Number of hidden layers in the attenuation network.
    W : int
        Hidden width.
    skips : list of int
        Layer indices that concatenate encoded points again.
    input_dims : dict
        Input dims for 'pts', 'view', and 'tx'.
    multires : dict
        Frequency-band counts for positional encoding.
    is_embeded : dict
        Whether to apply positional encoding per field.
    attn_output_dims : int
        Attenuation head size (amp, phase).
    sig_output_dims : int
        Signal head size (amp, phase).
    """

    def __init__(self, D=8, W=256, skips=[4],
                 input_dims={'pts': 3, 'view': 3, 'tx': 32},
                 multires={'pts': 10, 'view': 10, 'tx': 10},
                 is_embeded={'pts': True, 'view': True, 'tx': False},
                 attn_output_dims=2, sig_output_dims=2):
        super().__init__()
        self.skips = skips

        self.embed_pts_fn, input_pts_dim = get_embedder(
            multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_view_fn, input_view_dim = get_embedder(
            multires['view'], is_embeded['view'], input_dims['view'])
        self.embed_tx_fn, input_tx_dim = get_embedder(
            multires['tx'], is_embeded['tx'], input_dims['tx'])

        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )

        self.signal_linears = nn.ModuleList(
            [nn.Linear(input_view_dim + input_tx_dim + W, W)] +
            [nn.Linear(W, W)] * (D - 1) +
            [nn.Linear(W, W // 2)]
        )

        self.attenuation_output = nn.Linear(W, attn_output_dims)
        self.feature_layer = nn.Linear(W, W)
        self.signal_output = nn.Linear(W // 2, sig_output_dims)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, view, tx):
        """Predict attenuation and signal at sample points.

        Parameters
        ----------
        pts : torch.Tensor
            [..., 3] sample positions.
        view : torch.Tensor
            [..., 3] view directions.
        tx : torch.Tensor
            [..., tx_dim] transmitter / position features.

        Returns
        -------
        torch.Tensor
            [..., 4] (attn_amp, attn_phase, signal_amp, signal_phase).
        """
        pts = self.embed_pts_fn(pts).contiguous()
        view = self.embed_view_fn(view).contiguous()
        tx = self.embed_tx_fn(tx).contiguous()
        shape = pts.shape
        pts = pts.view(-1, list(pts.shape)[-1])
        view = view.view(-1, list(view.shape)[-1])
        tx = tx.view(-1, list(tx.shape)[-1])

        weight_dtype = self.attenuation_linears[0].weight.dtype
        pts = pts.to(dtype=weight_dtype)
        view = view.to(dtype=weight_dtype)
        tx = tx.to(dtype=weight_dtype)
        x = pts
        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts, x], -1)
        attn = self.attenuation_output(x)
        feature = self.feature_layer(x)
        x = torch.cat([feature, view, tx], -1)

        for i, layer in enumerate(self.signal_linears):
            x = F.relu(layer(x))
        signal = self.signal_output(x)

        outputs = torch.cat([attn, signal], -1).contiguous()
        return outputs.view(shape[:-1] + outputs.shape[-1:])
