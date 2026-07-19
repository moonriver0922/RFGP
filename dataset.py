# -*- coding: utf-8 -*-
"""Spectrum pretraining dataset loaders.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

# Feature layout constants (last dim of each sample).
SPT_DIM = 9 * 36  # 324
GATEWAY_DIM = 7
SCENARIO_DIM = 1
# Real data: no timestamp.
FEATURE_DIM = SCENARIO_DIM + GATEWAY_DIM + SPT_DIM  # 332

FEATURE_DIM_WITH_TS = 1 + FEATURE_DIM  # 333


def _list_tensor_files(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    return sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.endswith('.t') and os.path.isfile(os.path.join(root, name))
    )


def _as_int_scene_ids(values: torch.Tensor) -> List[int]:
    uniq = torch.unique(values.reshape(-1)).detach().cpu().tolist()
    return sorted(int(v) for v in uniq)


def _split_feature_channels(
    feat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split last-dim features into ``(ts, scenario, gateways, spectrum)``.

    Supports both 332-d (real, no ts) and 333-d (legacy with ts). For 332-d
    inputs, ``ts`` is a zeros tensor matching the remaining field shapes so
    callers can keep a stable API.
    """
    last = feat.shape[-1]
    if last == FEATURE_DIM:
        # scenario | gateways | spectrum
        scenario = feat[..., 0:1]
        gateways = feat[..., 1:1 + GATEWAY_DIM]
        spectrum = feat[..., 1 + GATEWAY_DIM:]
        ts = torch.zeros_like(scenario)
    elif last == FEATURE_DIM_WITH_TS:
        # ts | scenario | gateways | spectrum
        ts = feat[..., 0:1]
        scenario = feat[..., 1:2]
        gateways = feat[..., 2:2 + GATEWAY_DIM]
        spectrum = feat[..., 2 + GATEWAY_DIM:]
    else:
        raise ValueError(
            f"Expected last feature dim {FEATURE_DIM} (no ts) or "
            f"{FEATURE_DIM_WITH_TS} (legacy with ts), got {last}. "
            f"Layout should be "
            f"[scenario(1)|gateways({GATEWAY_DIM})|spectrum({SPT_DIM})] "
            f"or the same with a leading ts(1)."
        )
    if spectrum.shape[-1] != SPT_DIM:
        raise ValueError(
            f"Spectrum dim mismatch: expected {SPT_DIM}, got {spectrum.shape[-1]}")
    return ts, scenario, gateways, spectrum


class MyDataset(Dataset):
    """Load all spectrum shards from one or more modality directories.

    Parameters
    ----------
    data_roots : dict[str, str] or sequence of str
        Mapping ``modality -> directory`` (e.g. ble/wifi/iiot/rfid), or a list
        of directories. Every ``*.t`` file in each directory is loaded.
    chunksize : int
        Number of samples grouped per training chunk.
    """

    def __init__(
        self,
        data_roots: Union[Dict[str, str], Sequence[str]],
        chunksize: int,
    ):
        if isinstance(data_roots, dict):
            root_items = [(str(k), v) for k, v in data_roots.items()]
        else:
            root_items = [(f'source_{i}', root) for i, root in enumerate(data_roots)]

        self.modalities = [name for name, _ in root_items]
        shards: List[torch.Tensor] = []
        self.loaded_files: List[str] = []
        self.modality_stats: Dict[str, dict] = {}

        for modality, root in root_items:
            mod_files: List[str] = []
            mod_scenes = set()
            mod_chunks = 0
            files = _list_tensor_files(root)
            for file_path in files:
                try:
                    tensor = torch.load(file_path, map_location=torch.device('cpu'))
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load spectrum tensor '{file_path}': {exc}"
                    ) from exc
                # Real datasets may be stored as (N, F) without seq_len.
                # Normalize to the expected (N, n_seq, F) layout.
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(1)
                if tensor.ndim < 2:
                    raise ValueError(
                        f"Expected at least 2D tensor in '{file_path}', got shape {tuple(tensor.shape)}"
                    )
                feat_dim = tensor.shape[-1]
                if feat_dim not in (FEATURE_DIM, FEATURE_DIM_WITH_TS):
                    raise ValueError(
                        f"Unsupported feature dim {feat_dim} in '{file_path}'. "
                        f"Expected {FEATURE_DIM} (no ts) or "
                        f"{FEATURE_DIM_WITH_TS} (legacy with ts)."
                    )
                lens = tensor.shape[0]
                seq_len = tensor.shape[1]
                usable = lens // chunksize * chunksize
                if usable == 0:
                    continue
                tensor = tensor[:usable].reshape(
                    -1, chunksize, seq_len, feat_dim)
                # Scenario is at index 0 (332) or 1 (333).
                scenario_slice = (
                    tensor[..., 0:1] if feat_dim == FEATURE_DIM else tensor[..., 1:2])
                scene_ids = _as_int_scene_ids(scenario_slice)
                mod_scenes.update(scene_ids)
                mod_chunks += tensor.shape[0]
                shards.append(tensor)
                self.loaded_files.append(file_path)
                mod_files.append(file_path)

            self.modality_stats[modality] = {
                'root': root,
                'files': len(mod_files),
                'file_paths': mod_files,
                'scenes': sorted(mod_scenes),
                'num_scenes': len(mod_scenes),
                'chunks': mod_chunks,
                'exists': bool(root) and os.path.isdir(root) if root else False,
            }

        if not shards:
            roots = [root for _, root in root_items]
            raise FileNotFoundError(
                f"No usable spectrum tensors (*.t) found under: {roots}. "
                "Point path.train_data_roots / path.test_data_roots at directories "
                "that contain .t files."
            )

        # Mix of 332/333 shards is allowed; normalize by splitting channels.
        ts_list, scen_list, gw_list, spt_list = [], [], [], []
        for shard in shards:
            ts, scen, gw, spt = _split_feature_channels(shard)
            ts_list.append(ts)
            scen_list.append(scen)
            gw_list.append(gw)
            spt_list.append(spt)

        self.ts = torch.cat(ts_list, dim=0)
        self.scenario_info = torch.cat(scen_list, dim=0)
        self.gateways_info = torch.cat(gw_list, dim=0)
        self.spt = torch.cat(spt_list, dim=0)
        self.labels = self.spt
        # Keep a 332-d packed view for length / indexing: scenario|gw|spt.
        self.all_data = torch.cat(
            [self.scenario_info, self.gateways_info, self.spt], dim=-1)
        self.feature_dim = FEATURE_DIM
        self.scene_ids = _as_int_scene_ids(self.scenario_info)
        # One scene id per chunk (chunks are built from a single shard / scene)
        self.chunk_scene_ids = self.scenario_info[:, 0, 0, 0].long().contiguous()
        self._scene_to_indices = self._build_scene_index()

    def _build_scene_index(self) -> Dict[int, torch.Tensor]:
        index = {}
        for sid in self.scene_ids:
            index[sid] = (self.chunk_scene_ids == sid).nonzero(as_tuple=False).view(-1)
        return index

    def sample_chunk_indices(
        self,
        fraction: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample chunk indices with an independent quota per scene.

        Parameters
        ----------
        fraction : float
            Fraction of chunks to keep for each scene in ``(0, 1]``.
            ``1.0`` returns all chunks. Each scene keeps at least one chunk
            when it has data.
        generator : torch.Generator, optional
            RNG for reproducible draws.

        Returns
        -------
        torch.Tensor
            1-D int64 chunk indices into ``all_data`` / ``spt`` / etc.
        """
        fraction = float(fraction)
        if fraction <= 0:
            raise ValueError(f"data_fraction must be > 0, got {fraction}")
        if fraction >= 1.0:
            return torch.arange(len(self.all_data), dtype=torch.long)

        selected = []
        for sid in self.scene_ids:
            idx = self._scene_to_indices[sid]
            n = idx.numel()
            k = max(1, int(round(n * fraction)))
            k = min(k, n)
            perm = torch.randperm(n, generator=generator)[:k]
            selected.append(idx[perm])
        return torch.cat(selected, dim=0)

    def subset_report(self, indices: torch.Tensor, fraction: float, split: str = 'train') -> str:
        """Summarize how many chunks were kept per scene after sampling."""
        kept_scenes = self.chunk_scene_ids[indices]
        lines = [
            f"[{split}] active subset fraction={fraction:.4g}: "
            f"{indices.numel()}/{len(self.all_data)} chunk(s)"
        ]
        for sid in self.scene_ids:
            total = self._scene_to_indices[sid].numel()
            kept = int((kept_scenes == sid).sum().item())
            lines.append(f"  - scene {sid}: {kept}/{total} chunks")
        return '\n'.join(lines)

    def gather(self, indices: torch.Tensor):
        """Return (enc, spt, dec, labels, ts, scenario, gateways) for ``indices``."""
        n = indices.numel()
        enc = torch.arange(n).unsqueeze(1)
        dec = enc.clone()
        return (
            enc,
            self.spt[indices],
            dec,
            self.labels[indices],
            self.ts[indices],
            self.scenario_info[indices],
            self.gateways_info[indices],
        )

    def modality_report(self, split: str = 'train') -> str:
        """Human-readable per-modality load summary (full pool)."""
        lines = [f"[{split}] loaded {len(self.scene_ids)} scene(s) total "
                 f"from {len(self.loaded_files)} file(s), "
                 f"{len(self.all_data)} chunk(s):"]
        for modality in self.modalities:
            st = self.modality_stats[modality]
            if st['files'] == 0:
                status = 'missing dir' if not st['exists'] else '0 files'
                lines.append(
                    f"  - {modality}: 0 scenes ({status})  root={st['root']}")
            else:
                lines.append(
                    f"  - {modality}: {st['num_scenes']} scenes "
                    f"{st['scenes']}  files={st['files']}  "
                    f"chunks={st['chunks']}  root={st['root']}")
        lines.append(f"  - all scenes: {self.scene_ids}")
        return '\n'.join(lines)

    def loaddata(self):
        """Return full-pool tensors (no subsetting)."""
        n = len(self.all_data)
        enc = torch.arange(n).unsqueeze(1)
        return (
            enc,
            self.spt,
            enc.clone(),
            self.labels,
            self.ts,
            self.scenario_info,
            self.gateways_info,
        )


def discover_scene_ids(*datasets: Optional[MyDataset]) -> List[int]:
    """Union of scene ids across datasets, sorted."""
    ids = set()
    for ds in datasets:
        if ds is None:
            continue
        ids.update(ds.scene_ids)
    return sorted(ids)
