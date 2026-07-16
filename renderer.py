# -*- coding: utf-8 -*-
"""Ray marching and wireless signal rendering (spectrum / RSSI / CSI)."""
import logging

import numpy as np
import scipy.constants as sc
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


def _ones_like_leading(alpha: torch.Tensor) -> torch.Tensor:
    """Ones column on the same device/dtype as alpha's batch axis."""
    return torch.ones((alpha.shape[0], 1), device=alpha.device, dtype=alpha.dtype)


def _large_dists(ref: torch.Tensor) -> torch.Tensor:
    """Trailing large distance sample matching ref's leading dims."""
    return torch.full(ref[..., :1].shape, 1e10, device=ref.device, dtype=ref.dtype)


class Renderer:
    """Base sampler along rays for NeRF2 rendering."""

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        ----------
        networks_fn : callable
            NeRF2-like network mapping (pts, view, tx) -> raw features.
        near : float
            Near bound of ray samples.
        far : float
            Far bound of ray samples.
        n_samples : int
            Number of samples per ray.
        """
        self.network_fn = networks_fn
        self.n_samples = kwargs['n_samples']
        self.near = kwargs['near']
        self.far = kwargs['far']

    def sample_points(self, rays_o, rays_d):
        """Sample points along rays.

        Parameters
        ----------
        rays_o : torch.Tensor
            [n_rays, 3] ray origins.
        rays_d : torch.Tensor
            [n_rays, 3] ray directions.

        Returns
        -------
        pts : torch.Tensor
            [n_rays, n_samples, 3] sampled points.
        t_vals : torch.Tensor
            [n_rays, n_samples] distances from origin.
        """
        shape = list(rays_o.shape)
        shape[-1] = 1
        near, far = torch.full(shape, self.near), torch.full(shape, self.far)
        t_vals = torch.linspace(0., 1., steps=self.n_samples) * (far - near) + near
        t_vals = t_vals.to(rays_o.device)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]
        return pts, t_vals


class Renderer_spectrum(Renderer):
    """Renderer for directional spectrum (integral along a single ray)."""

    def __init__(self, networks_fn, **kwargs) -> None:
        super().__init__(networks_fn, **kwargs)

    def render_ss(self, tx, rays_o, rays_d):
        """Render signal strength for each ray.

        Parameters
        ----------
        tx : torch.Tensor
            [batchsize, tx_dim] transmitter / position features.
        rays_o : torch.Tensor
            [batchsize, 3] ray origins.
        rays_d : torch.Tensor
            [batchsize, 3] ray directions.

        Returns
        -------
        torch.Tensor
            [batchsize] absolute received signal strength per ray.
        """
        pts, t_vals = self.sample_points(rays_o, rays_d)
        view = rays_d[:, None].expand(pts.shape)
        tx = tx.unsqueeze(1).repeat(1, pts.shape[1], 1)
        raw = self.network_fn(pts, view, tx)
        receive_ss = self.raw2outputs(raw, t_vals, rays_d)
        return receive_ss

    def raw2outputs(self, raw, r_vals, rays_d):
        """Convert network outputs to per-ray absolute signal strength.

        Parameters
        ----------
        raw : torch.Tensor
            [batchsize, n_samples, 4] model predictions.
        r_vals : torch.Tensor
            [batchsize, n_samples] integration distances.
        rays_d : torch.Tensor
            [batchsize, 3] ray directions.

        Returns
        -------
        torch.Tensor
            [batchsize] abs(received signal) per ray.
        """
        raw2alpha = lambda raw, dists: 1. - torch.exp(-raw * dists)
        raw2phase = lambda raw, dists: raw * dists

        dists = r_vals[..., 1:] - r_vals[..., :-1]
        dists = torch.cat([dists, _large_dists(dists)], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        att_a, att_p, s_a, s_p = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
        att_p, s_p = torch.sigmoid(att_p) * np.pi * 2, torch.sigmoid(s_p) * np.pi * 2
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        alpha = raw2alpha(att_a, dists)
        phase = raw2phase(att_p, dists)

        att_i = torch.cumprod(
            torch.cat([_ones_like_leading(alpha), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        path = torch.cat([r_vals[..., 1:], _large_dists(r_vals)], -1)
        path_loss = 0.025 / path
        phase_i = torch.cumsum(
            torch.cat([_ones_like_leading(alpha), phase], -1), -1)[:, :-1]
        phase_i = torch.exp(1j * phase_i)

        if torch.isnan(phase_i).any() or torch.isnan(att_i).any() or torch.isnan(path_loss).any() \
                or torch.isnan(s_a).any() or torch.isnan(s_p).any():
            logger.warning("NaN detected in spectrum rendering intermediates")

        receive_signal = torch.sum(s_a * torch.exp(1j * s_p) * att_i * phase_i * path_loss, -1)
        receive_signal = abs(receive_signal)
        return receive_signal


class Renderer_RSSI(Renderer):
    """Renderer for RSSI (integral over all directions)."""

    def __init__(self, networks_fn, **kwargs) -> None:
        super().__init__(networks_fn, **kwargs)

    def render_rssi(self, tx, rays_o, rays_d):
        """Render RSSI for each gateway (chunked over directions).

        Parameters
        ----------
        tx : torch.Tensor
            [batchsize, 3] transmitter positions.
        rays_o : torch.Tensor
            [batchsize, 3] ray origins.
        rays_d : torch.Tensor
            [batchsize, 9x36x3] flattened ray directions.
        """
        batchsize, _ = tx.shape
        rays_d = torch.reshape(rays_d, (batchsize, -1, 3))
        chunks = 36
        chunks_num = 36 // chunks
        rays_o_chunk = rays_o.expand(chunks, -1, -1).permute(1, 0, 2)
        tags_chunk = tx.expand(chunks, -1, -1).permute(1, 0, 2)
        recv_signal = torch.zeros(batchsize, device=tx.device, dtype=tx.dtype)
        for i in range(chunks_num):
            rays_d_chunk = rays_d[:, i * chunks:(i + 1) * chunks, :]
            pts, t_vals = self.sample_points(rays_o_chunk, rays_d_chunk)
            views_chunk = rays_d_chunk[..., None, :].expand(pts.shape)
            tx_chunk = tags_chunk[..., None, :].expand(pts.shape)
            raw = self.network_fn(pts, views_chunk, tx_chunk)
            recv_signal_chunks = self.raw2outputs_signal(raw, t_vals, rays_d_chunk)
            recv_signal += recv_signal_chunks
        return recv_signal

    def raw2outputs_signal(self, raw, r_vals, rays_d):
        """Integrate complex RSSI along rays then over directions."""
        wavelength = sc.c / 2.4e9
        raw2phase = lambda raw, dists: raw + 2 * np.pi * dists / wavelength
        raw2amp = lambda raw, dists: -raw * dists

        dists = r_vals[..., 1:] - r_vals[..., :-1]
        dists = torch.cat([dists, _large_dists(dists)], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        att_a, att_p, s_a, s_p = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
        att_p, s_p = torch.sigmoid(att_p) * np.pi * 2 - np.pi, torch.sigmoid(s_p) * np.pi * 2 - np.pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        amp = raw2amp(att_a, dists)
        phase = raw2phase(att_p, dists)

        amp_i = torch.exp(torch.cumsum(amp, -1))
        phase_i = torch.exp(1j * torch.cumsum(phase, -1))

        recv_signal = torch.sum(s_a * torch.exp(1j * s_p) * amp_i * phase_i, -1)
        recv_signal = torch.sum(recv_signal, -1)
        return abs(recv_signal)


class Renderer_CSI(Renderer):
    """Renderer for CSI (OFDM subcarriers, integral over directions)."""

    def __init__(self, networks_fn, **kwargs) -> None:
        super().__init__(networks_fn, **kwargs)

    def render_csi(self, uplink, rays_o, rays_d):
        """Render downlink CSI given uplink CSI and ray geometry.

        Parameters
        ----------
        uplink : torch.Tensor
            [batchsize, 52] uplink CSI (26 real + 26 imag).
        rays_o : torch.Tensor
            [batchsize, 3] ray origins.
        rays_d : torch.Tensor
            [batchsize, 9x36x3] flattened ray directions.
        """
        rays_d = rearrange(rays_d, 'b (v d) -> b v d', d=3)
        batchsize, viewsize, _ = rays_d.shape
        rays_o = repeat(rays_o, 'b d -> b v d', v=viewsize)
        uplink = repeat(uplink, 'b d -> b v d', v=viewsize)

        pts, t_vals = self.sample_points(rays_o, rays_d)
        views = repeat(rays_d, 'b v d -> b v p d', p=self.n_samples)
        uplink = repeat(uplink, 'b v d -> b v p d', p=self.n_samples)

        raw = self.network_fn(pts, views, uplink)
        recv_signal = self.raw2outputs_signal(raw, t_vals, rays_d)
        return recv_signal

    def raw2outputs_signal(self, raw, r_vals, rays_d):
        """Integrate complex OFDM CSI along rays then over directions."""
        wavelength = sc.c / 2.4e9
        raw2phase = lambda raw, dists: raw + 2 * np.pi * dists / wavelength
        raw2amp = lambda raw, dists: -raw * dists

        dists = r_vals[..., 1:] - r_vals[..., :-1]
        dists = torch.cat([dists, _large_dists(dists)], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        att_a, att_p, s_a, s_p = raw[..., :26], raw[..., 26:52], raw[..., 52:78], raw[..., 78:104]
        att_p, s_p = torch.sigmoid(att_p) * np.pi * 2 - np.pi, torch.sigmoid(s_p) * np.pi * 2 - np.pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        dists = dists.unsqueeze(-1)

        amp = raw2amp(att_a, dists)
        phase = raw2phase(att_p, dists)

        amp_i = torch.exp(torch.cumsum(amp, -2))
        phase_i = torch.exp(1j * torch.cumsum(phase, -2))

        recv_signal = torch.sum(s_a * torch.exp(1j * s_p) * amp_i * phase_i, -2)
        recv_signal = torch.sum(recv_signal, 1)
        return recv_signal


renderer_dict = {
    "spectrum": Renderer_spectrum,
    "rssi": Renderer_RSSI,
    "csi": Renderer_CSI,
}
