# vjepa_probe.py
from __future__ import annotations
import os, re, csv, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor, infer_device

# ==========
# Labels
# ==========
@dataclass(frozen=True)
class SplitKey:
    split: str          # "train" | "val"
    is_dark: str        # "True" | "False"
    turn_dir3: str      # "left" | "right" | "straight"

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.split, self.is_dark, self.turn_dir3)

    def __str__(self) -> str:
        return f"{self.split}|is_dark={self.is_dark}|turn={self.turn_dir3}"

def identity_label(k: SplitKey): return k.as_tuple()
def label_turn_only(k: SplitKey): return k.turn_dir3
def label_is_dark_only(k: SplitKey): return k.is_dark

# ==========
# Discovery
# ==========
def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# replace discover_splits() with this version
def discover_splits(root: str):
    """
    Returns:
      index: { SplitKey -> [sequence_dir, ...] }  # sequence_dir = .../sequence_XXXX
      seq_frames: { (sequence_dir, modality) -> [frame_path, ...] }  # modality in {"gen","real"}
    """
    root = Path(root)
    index: Dict[SplitKey, List[Path]] = {}
    seq_frames: Dict[Tuple[Path, str], List[Path]] = {}

    def natural_key(s: str):
        import re
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for dark_dir in sorted(split_dir.glob("is_dark=*")):
            if not dark_dir.is_dir():
                continue
            is_dark = dark_dir.name.split("=")[-1]
            for turn_dir in sorted(dark_dir.glob("turn_dir3=*")):
                if not turn_dir.is_dir():
                    continue
                turn = turn_dir.name.split("=")[-1]
                key = SplitKey(split, is_dark, turn)
                seq_dirs = [p for p in sorted(turn_dir.glob("sequence_*")) if p.is_dir()]
                keep = []
                for sd in seq_dirs:
                    # look for sd/gen and sd/real
                    for modality in ("gen", "real"):
                        mdir = sd / modality
                        if not mdir.is_dir():
                            continue
                        frames = sorted(
                            [*mdir.glob("*.jpg"), *mdir.glob("*.jpeg"), *mdir.glob("*.png")],
                            key=lambda p: natural_key(p.name)
                        )
                        if frames:
                            seq_frames[(sd, modality)] = frames
                            keep.append(sd)
                keep = sorted(set(keep))
                index[key] = keep
    return index, seq_frames


def summarize_index(index, seq_frames) -> str:
    lines, total = [], 0
    for k in sorted(index.keys(), key=lambda x: (x.split, x.is_dark, x.turn_dir3)):
        nseq = len(index[k]); total += nseq
        nframes = sum(len(seq_frames[sd]) for sd in index[k])
        lines.append(f"{k}: {nseq} sequences, {nframes} frames")
    lines.append(f"TOTAL sequences: {total}")
    return "\n".join(lines)

# ==========
# Dataset
# ==========
class SequenceDataset(Dataset):
    def __init__(
        self,
        index,
        seq_frames,
        include_keys=None,
        exclude_keys=None,
        num_frames: int = 20,
        sample_mode: str = "uniform",
        modality: str = "real",                # NEW: "gen" or "real"
        exclude_same_seq_with: Optional[set] = None,  # NEW: avoid leakage
    ):
        self.seq_frames = seq_frames
        self.num_frames = num_frames
        self.sample_mode = sample_mode
        self.modality = modality
        self.exclude_same_seq_with = exclude_same_seq_with or set()

        pool = []
        for k, seqs in index.items():
            if exclude_keys and k in exclude_keys:
                continue
            if include_keys is not None and k not in include_keys:
                continue
            for sd in seqs:
                if (str(sd), modality) in self.exclude_same_seq_with:
                    continue
                # only include if we actually have frames for that modality
                if (sd, modality) in self.seq_frames:
                    pool.append((k, sd))
        self.items = pool

    def __len__(self): return len(self.items)

    def _choose_indices(self, n, T):
        if n == 0: return []
        if self.sample_mode == "first":
            idx = list(range(min(T, n)))
            return idx + [idx[-1]] * (T - len(idx)) if len(idx) < T else idx
        if self.sample_mode == "middle" and n >= T:
            start = (n - T) // 2
            return list(range(start, start + T))
        if self.sample_mode == "last":
            idx = list(range(max(0, n - T), n))
            return [idx[0]] * (T - len(idx)) + idx if len(idx) < T else idx
        if n >= T:
            return [round(i*(n-1)/(T-1)) for i in range(T)]
        base = list(range(n))
        return base + [base[-1]] * (T - n)

    def __getitem__(self, i):
        key, seq_dir = self.items[i]
        frames = self.seq_frames[(seq_dir, self.modality)]
        idxs = self._choose_indices(len(frames), self.num_frames)
        imgs = [Image.open(str(frames[j])).convert("RGB") for j in idxs]
        return {
            "images": imgs,
            "key": key,
            "seq_dir": str(seq_dir),
            "modality": self.modality,
        }

# ==========
# Features (GPU, cache)
# ==========
class FeatureExtractor:
    def __init__(self, hf_repo="facebook/vjepa2-vith-fpc64-256", cache_dir="./vjepa_cache", fp16=True, crop=(354, 448)):
        self.device = infer_device()
        dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else None
        self.model = AutoModel.from_pretrained(hf_repo, torch_dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.crop = crop
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


    def _hash(self, seq_dir: str, T: int, modality: str) -> str:
        import hashlib
        return hashlib.sha1(f"{seq_dir}|{T}|{modality}".encode()).hexdigest()[:16]


    def encode_dataset(self, ds: SequenceDataset, batch_size=4, num_workers=4, use_cache=True):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda b: b)
        X_all, labels, seqs, modalities = [], [], [], []
        for pack in loader:
            cache_hits, to_encode, to_idx = [], [], []
            for j, item in enumerate(pack):
                labels.append(item["key"])
                seqs.append(item["seq_dir"])
                modalities.append(item["modality"])
                vec = None
                if self.cache_dir:
                    h = self._hash(item["seq_dir"], len(item["images"]), item["modality"])
                    f = self.cache_dir / f"{h}.pt"
                    if use_cache and f.exists():
                        vec = torch.load(f, map_location="cpu")
                cache_hits.append(vec)
            for j, item in enumerate(pack):
                if cache_hits[j] is None:
                    to_encode.append(item["images"])
                    to_idx.append(j)
            if to_encode:
                enc = self.encode_batch(to_encode)
                m = 0
                for j in to_idx:
                    cache_hits[j] = enc[m]
                    if self.cache_dir:
                        h = self._hash(pack[j]["seq_dir"], len(pack[j]["images"]), pack[j]["modality"])
                        torch.save(enc[m], self.cache_dir / f"{h}.pt")
                    m += 1
            X_all.append(torch.stack([cache_hits[j] for j in range(len(pack))], dim=0))
        X = torch.cat(X_all, dim=0) if X_all else torch.empty(0, 1)
        return X, labels, seqs, modalities


    @torch.no_grad()
    def encode_batch(self, samples: List[List[Image.Image]]) -> torch.Tensor:
        feats = []
        for imgs in samples:
            proc = self.processor(imgs, return_tensors="pt", do_center_crop=True, crop_size={"height": self.crop[0], "width": self.crop[1]})
            x = proc["pixel_values_videos"].to(self.device)        # [1, T, C, H, W]
            out = self.model(pixel_values_videos=x)
            enc = out.last_hidden_state

            B, _, D = enc.shape
            T = x.shape[1] //2 # actual number of frames input
            S = enc.shape[1] // T
            _, _, _, H, W = x.shape
            grid_h, grid_w = H // 16, W // 16

            enc = enc.reshape(B, T, grid_h, grid_w, D).permute(0, 1, 4, 2, 3)
            enc = enc.reshape(B * T, D, grid_h, grid_w)
            enc = F.adaptive_avg_pool2d(enc, (4, 6))
            enc = enc.reshape(B, T, D, 4, 6)
            enc = enc.permute(0, 3, 4, 2, 1).reshape(B, 24 * D, T)
            
            if T >= 2:
                enc = F.avg_pool1d(enc, kernel_size=2, stride=2, ceil_mode=True)

            enc = enc.reshape(B, -1)
            feats.append(enc.squeeze(0).float().cpu())
        return torch.stack(feats, dim=0)


# ==========
# Distance & kNN
# ==========
def l2_normalize(X: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return X / (X.norm(dim=1, keepdim=True) + eps)

def build_gallery_by_slice(X, labels, seqs, modalities):
    feats_by: Dict[SplitKey, List[torch.Tensor]] = {}
    seqs_by:  Dict[SplitKey, List[str]] = {}
    mods_by:  Dict[SplitKey, List[str]] = {}
    for xi, lab, sp, md in zip(X, labels, seqs, modalities):
        feats_by.setdefault(lab, []).append(xi.unsqueeze(0))
        seqs_by.setdefault(lab, []).append(sp)
        mods_by.setdefault(lab, []).append(md)
    feats_by = {k: torch.cat(v, dim=0) for k, v in feats_by.items()}
    return feats_by, seqs_by, mods_by

def concat_gallery(G_by, seqs_by, mods_by):
    keys = sorted(G_by.keys(), key=lambda k: (k.split, k.is_dark, k.turn_dir3))
    parts, seq_parts, mod_parts, ranges = [], [], [], []
    start = 0
    for k in keys:
        X = G_by[k]
        parts.append(X)
        seq_parts += seqs_by[k]
        mod_parts += mods_by[k]
        end = start + len(X)
        ranges.append((start, end))
        start = end
    G_all = torch.cat(parts, dim=0) if parts else torch.empty(0, next(iter(G_by.values())).shape[1])
    return G_all, keys, ranges, seq_parts, mod_parts

def knn_predict(Q, G_by, seqs_by, mods_by, k=5, label_project=identity_label):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Qn = l2_normalize(Q).to(device)
    G_all, keys, ranges, seq_all, mod_all = concat_gallery(G_by, seqs_by, mods_by)
    Gn = l2_normalize(G_all).to(device)
    with torch.no_grad():
        sims = Qn @ Gn.T
        k_eff = min(k, Gn.shape[0]) if Gn.numel() > 0 else 1
        topk = torch.topk(sims, k=k_eff, dim=1, largest=True, sorted=True)
        idx = topk.indices.cpu()

    idx2key: List[SplitKey] = [None] * Gn.shape[0]  # type: ignore
    for key, (s, e) in zip(keys, ranges):
        for i in range(s, e): idx2key[i] = key

    preds, nn_indices, nn_splits, nn_seqs, nn_modalities = [], [], [], [], []
    for i in range(idx.shape[0]):
        inds = idx[i].tolist()
        labs = [idx2key[j] for j in inds]
        seqs = [seq_all[j] for j in inds]
        mods = [mod_all[j] for j in inds]
        proj = [label_project(l) for l in labs]
        counts: Dict[Any, int] = {}
        for p in proj: counts[p] = counts.get(p, 0) + 1
        pred = max(counts.items(), key=lambda kv: kv[1])[0]
        preds.append(pred); nn_indices.append(inds); nn_splits.append(labs); nn_seqs.append(seqs); nn_modalities.append(mods)
    return preds, nn_indices, nn_splits, nn_seqs, nn_modalities

def compute_distance_cube(Q: torch.Tensor, G_by: Dict[SplitKey, torch.Tensor]) -> Dict[SplitKey, torch.Tensor]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Qn = l2_normalize(Q).to(device)
    out: Dict[SplitKey, torch.Tensor] = {}
    with torch.no_grad():
        for k, G in G_by.items():
            Gn = l2_normalize(G).to(device)
            sims = Qn @ Gn.T
            out[k] = (1.0 - sims).cpu()  # [Nq, Ng]
    return out


def accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))

def per_slice_nearest_share(Q: torch.Tensor, G_by: Dict[SplitKey, torch.Tensor]) -> Dict[SplitKey, float]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Qn = l2_normalize(Q).to(device)
    total = Q.shape[0]
    counts: Dict[SplitKey, int] = {k: 0 for k in G_by.keys()}
    with torch.no_grad():
        best_sims = torch.full((Q.shape[0],), -1e9, device=device)
        best_keys: List[Optional[SplitKey]] = [None] * Q.shape[0]
        for k, G in G_by.items():
            Gn = l2_normalize(G).to(device)
            sims = Qn @ Gn.T
            max_sims, _ = sims.max(dim=1)
            better = max_sims > best_sims
            best_sims = torch.where(better, max_sims, best_sims)
            for i in range(Q.shape[0]):
                if better[i]: best_keys[i] = k
        for bk in best_keys:
            if bk is not None: counts[bk] += 1
    return {k: counts[k] / max(1, total) for k in counts}

# ==========
# Orchestration
# ==========
@dataclass
class RunConfig:
    root: str
    query_key: SplitKey
    gallery_mode: str = "others"        # "others" | "all" | "train_only" | "val_only"
    num_frames: int = 20
    sample_mode: str = "last"
    hf_repo: str = "facebook/vjepa2-vitl-fpc64-256"
    cache_dir: str = "./vjepa_cache"
    batch_size_enc: int = 4
    num_workers: int = 4
    k: int = 5
    label_space: str = "full"           # "full" | "turn" | "dark"
    out_dir: str = "./vjepa_out"

def pick_label_projector(name: str):
    if name == "full": return identity_label
    if name == "turn": return label_turn_only
    if name == "dark": return label_is_dark_only
    raise ValueError(f"Unknown label_space: {name}")

def select_gallery_keys(all_keys: List[SplitKey], q: SplitKey, mode: str) -> List[SplitKey]:
    if mode == "others":     return [k for k in all_keys if k != q]
    if mode == "all":        return all_keys
    if mode == "train_only": return [k for k in all_keys if k.split == "train"]
    if mode == "val_only":   return [k for k in all_keys if k.split == "val"]
    raise ValueError(mode)

def run_experiment(cfg: RunConfig):
    index, seq_frames = discover_splits(cfg.root)
    print(summarize_index(index, {sd: fr for (sd,_), fr in seq_frames.items()}))  # quick overview

    # To avoid leakage, record the set of (seq_dir, modality) used in query and exclude same seq@real if desired.
    exclude_gallery_pairs = set()

    # queries: generated from ONE split key
    q_ds = SequenceDataset(
        index, seq_frames,
        include_keys=[cfg.query_key],
        num_frames=cfg.num_frames, sample_mode=cfg.sample_mode,
        modality="gen",
    )
    # mark gen pairs to optionally exclude same sequence in real gallery
    for _, seq_dir in q_ds.items:
        exclude_gallery_pairs.add((str(seq_dir), "real"))

    # gallery: real from ALL splits (or follow gallery_mode if you want train_only, etc.)
    all_keys = sorted(index.keys(), key=lambda k: (k.split, k.is_dark, k.turn_dir3))
    gal_keys = select_gallery_keys(all_keys, cfg.query_key, "all")  # <-- real from all splits
    g_ds = SequenceDataset(
        index, seq_frames,
        include_keys=gal_keys,
        num_frames=cfg.num_frames, sample_mode=cfg.sample_mode,
        modality="real",
        #exclude_same_seq_with=exclude_gallery_pairs,   # avoid same sequence as its own nearest
    )

    fe = FeatureExtractor(cfg.hf_repo, cfg.cache_dir, fp16=True)
    Q, yq, seq_q, mod_q = fe.encode_dataset(q_ds, batch_size=cfg.batch_size_enc, num_workers=cfg.num_workers)
    G, yg, seq_g, mod_g = fe.encode_dataset(g_ds, batch_size=cfg.batch_size_enc, num_workers=cfg.num_workers)

    G_by, seqs_by, mods_by = build_gallery_by_slice(G, yg, seq_g, mod_g)
    D_by = compute_distance_cube(Q, G_by)

    projector = pick_label_projector(cfg.label_space)
    pred_labels, nn_idx, nn_splits, nn_seqs, nn_mods = knn_predict(Q, G_by, seqs_by, mods_by, k=cfg.k, label_project=projector)
    y_true = [projector(lbl) for lbl in yq]
    top1 = accuracy(y_true, pred_labels)
    share = per_slice_nearest_share(Q, G_by)

    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    export_distance_cube_csv(out / "distance_cube.csv", D_by, seq_q, yq, mod_q, seqs_by, mods_by)
    export_knn_csv(out / "knn_predictions.csv", seq_q, yq, mod_q, pred_labels, nn_splits, nn_seqs, nn_mods)

    summary_path = out / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["top1_acc", f"{top1:.6f}"])
        for sk in sorted(share.keys(), key=lambda x: (x.split, x.is_dark, x.turn_dir3)):
            writer.writerow([f"share:{sk}", f"{share[sk]:.6f}"])

    return {
        "query_key": str(cfg.query_key),
        "gallery_keys": [str(k) for k in gal_keys],
        "num_queries": len(Q),
        "num_gallery": len(G),
        "top1_acc": top1,
        "per_slice_share": {str(k): v for k, v in share.items()},
        "distance_cube_shapes": {str(k): tuple(t.shape) for k, t in D_by.items()},
        "csv_paths": {
            "distance_cube": str(out / "distance_cube.csv"),
            "knn_predictions": str(out / "knn_predictions.csv"),
            "summary": str(summary_path),
        }
    }

# ==========
# CSV Export
# ==========
def export_distance_cube_csv(path, D_by, query_seqs, query_labels, query_modalities, seqs_by, mods_by):
    with path.open("w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["query_seq","query_modality","query_split","query_is_dark","query_turn",
                    "gallery_split","gallery_is_dark","gallery_turn",
                    "gallery_seq","gallery_modality","distance"])
        keys = sorted(D_by.keys(), key=lambda k: (k.split, k.is_dark, k.turn_dir3))
        for qi, (qseq, qlab, qmod) in enumerate(zip(query_seqs, query_labels, query_modalities)):
            for k in keys:
                dmat = D_by[k]      # [Nq, Ng_k]
                row = dmat[qi]
                gseqs = seqs_by[k]; gmods = mods_by[k]
                for gj, dist in enumerate(row.tolist()):
                    w.writerow([qseq, qmod, qlab.split, qlab.is_dark, qlab.turn_dir3,
                                k.split, k.is_dark, k.turn_dir3,
                                gseqs[gj], gmods[gj], f"{dist:.6f}"])

def export_knn_csv(path, query_seqs, query_labels, query_modalities,
                   pred_labels, nn_splits, nn_seqs, nn_modalities):
    with path.open("w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["query_seq","query_modality","query_split","query_is_dark","query_turn",
                    "predicted_label",
                    "nearest_split","nearest_is_dark","nearest_turn","nearest_seq","nearest_modality",
                    "topk_splits","topk_seqs","topk_modalities"])
        for qseq, qlab, qmod, pred, splits_k, seqs_k, mods_k in zip(
            query_seqs, query_labels, query_modalities, pred_labels, nn_splits, nn_seqs, nn_modalities
        ):
            nk = splits_k[0] if splits_k else None
            ns = seqs_k[0] if seqs_k else ""
            nm = mods_k[0] if mods_k else ""
            w.writerow([
                qseq, qmod, qlab.split, qlab.is_dark, qlab.turn_dir3,
                str(pred),
                (nk.split if nk else ""), (nk.is_dark if nk else ""), (nk.turn_dir3 if nk else ""), ns, nm,
                "|".join(str(k) for k in splits_k),
                "|".join(seqs_k),
                "|".join(mods_k),
            ])
# ==========
# main() with fixed params
# ==========
def main():
    roots = [
        #"/work/dlclarge2/faridk-diff_force/orbis/logs_wm/2025-09-22T00-35-42_mg_vds_lmbgpu20_backup/gen_rollout/default_data/ep18iter118332_30steps",
       "/work/dlclarge2/faridk-diff_force/orbis/logs_wm/2025-09-22T07-42-12_van_vds_DLC24662470/gen_rollout/default_data/ep18iter135242_30steps",
    ]
    query_keys = [
        SplitKey("val", "True", "left"),
        SplitKey("val", "False", "right"),
    ]
    common_cfg = dict(
        gallery_mode="all",
        num_frames=10,
        sample_mode="last",
        hf_repo="facebook/vjepa2-vitl-fpc64-256",
        cache_dir="./vjepa_cache",
        batch_size_enc=16,
        num_workers=4,
        k=3,
        label_space="full",
    )
    base_out = Path("./vjepa_out")

    for root in roots:
        for key in query_keys:
            key_dir = f"{key.split}_dark-{key.is_dark}_turn-{key.turn_dir3}"
            out_dir = base_out / Path(root).name / key_dir
            cfg = RunConfig(
                root=root,
                query_key=key,
                out_dir=str(out_dir),
                **common_cfg,
            )
            print(f"\n=== Running root={root}, key={key} ===")
            res = run_experiment(cfg)
            print(f"  #Queries:       {res['num_queries']}")
            print(f"  #Gallery total: {res['num_gallery']}")
            print(f"  Top-1:          {res['top1_acc']:.4f}")
            print("  Per-slice nearest share:")
            for k, v in res["per_slice_share"].items():
                print(f"    {k}: {v:.3f}")
            print("  Distance cube shapes:")
            for k, s in res["distance_cube_shapes"].items():
                print(f"    {k}: {s}")
            print("  CSV exported:")
            for name, p in res["csv_paths"].items():
                print(f"    {name}: {p}")

if __name__ == "__main__":
    main()
