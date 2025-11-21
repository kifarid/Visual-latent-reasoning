"""
Create HDF5 Virtual Datasets (VDS) for train/intermixed/val/test splits based on a
labels parquet.

Given one or more H5 shards created by create_h5_flat_from_videos.py, and a parquet
file containing at least (video_name, frame_idx, split), this script builds a single
VDS H5 with four groups: /train, /intermixed, /val, /test. Each split contains:

- /frames [N, H, W, C] (uint8) as a VDS stacking slices from source shards
- /index/video_id [N] and /index/frame_idx [N] as VDS aligned 1:1 with frames
- /index/contiguous_runs [M, 4] listing (start, length, video_id, frame_start)

The value of the parquet ``split`` column (configurable) decides which group a frame
is assigned to. Any rows with other split values are ignored.

Example:
  python h5_dataloader/create_vds_splits.py \
    --parquet /path/to/labels.parquet \
    --h5-sources out/shards_0.h5 out/shards_1.h5 \
    --output out/vds_splits.h5
"""

from __future__ import annotations

import argparse
import os
import glob
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd


def _sanitize_path_component(value: object) -> str:
    """Return a string safe for use within an HDF5 path component."""

    text = str(value)
    return text.replace("/", "_")


def _format_subsplit_group_name(
    split_name: str, columns: Tuple[str, ...], values: Tuple[object, ...]
) -> str:
    parts = [str(split_name), "subsplits"]
    for col, val in zip(columns, values):
        parts.append(f"{col}={_sanitize_path_component(val)}")
    return "/".join(parts)


def _read_parquet_groups(
    parquet_path: str,
    video_col: str,
    frame_idx_col: str,
    split_col: str,
    allowed_splits: Iterable[str],
    subsplit_cols: Iterable[str] = (),
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Tuple[Tuple[str, str], ...]],
]:
    """Return mappings for primary splits and optional compositional subsplits.

    Only rows whose ``split_col`` value is in ``allowed_splits`` are considered.
    When ``subsplit_cols`` are provided, the function also aggregates frame indices
    for every unique combination of those columns per split.
    """
    splits = list(allowed_splits)
    split_set = set(splits)
    out: Dict[str, Dict[str, List[np.ndarray]]] = {k: defaultdict(list) for k in splits}

    subsplit_cols = tuple(dict.fromkeys(col for col in subsplit_cols if col and col != split_col))
    subsplit_collect: Dict[str, Dict[str, List[np.ndarray]]] = {}
    subsplit_meta: Dict[str, Tuple[Tuple[str, str], ...]] = {}

    try:
        needed_columns = {video_col, frame_idx_col, split_col, *subsplit_cols}
        df = pd.read_parquet(parquet_path, columns=list(needed_columns))
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet {parquet_path} with pandas. Error: {e}")

    df = df[df[split_col].isin(split_set)]
    for split_name, split_df in df.groupby(split_col):
        split_key = str(split_name)
        if split_key not in split_set:
            continue
        for vname, grp in split_df.groupby(video_col):
            out[split_key][str(vname)].append(grp[frame_idx_col].to_numpy(dtype=np.int64))

    out_final: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in splits}
    for split_name, per_video in out.items():
        for vname, arrays in per_video.items():
            if not arrays:
                continue
            arr = np.unique(np.concatenate(arrays).astype(np.int64))
            out_final[split_name][vname] = arr
    if not subsplit_cols:
        return out_final, {}, {}

    group_cols = [split_col, *subsplit_cols, video_col]
    for keys, grp in df.groupby(group_cols):
        split_value = str(keys[0])
        if split_value not in split_set:
            continue
        subsplit_values = tuple(str(v) for v in keys[1 : 1 + len(subsplit_cols)])
        video_value = str(keys[-1])
        group_name = _format_subsplit_group_name(split_value, subsplit_cols, subsplit_values)
        per_video = subsplit_collect.setdefault(group_name, defaultdict(list))
        per_video[video_value].append(grp[frame_idx_col].to_numpy(dtype=np.int64))
        if group_name not in subsplit_meta:
            subsplit_meta[group_name] = tuple(
                (col, val) for col, val in zip(subsplit_cols, subsplit_values)
            )

    subsplit_final: Dict[str, Dict[str, np.ndarray]] = {}
    for group_name, per_video in subsplit_collect.items():
        subsplit_final[group_name] = {}
        for vname, arrays in per_video.items():
            if not arrays:
                continue
            arr = np.unique(np.concatenate(arrays).astype(np.int64))
            subsplit_final[group_name][vname] = arr

    return out_final, subsplit_final, subsplit_meta


def _collect_sources(paths: Iterable[str], directory: str = "") -> List[str]:
    srcs: List[str] = []
    for p in paths:
        if not p:
            continue
        if not os.path.exists(p):
            raise FileNotFoundError(f"H5 source not found: {p}")
        srcs.append(p)
    if directory:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"H5 directory not found: {directory}")
        dir_sources = sorted(glob.glob(os.path.join(directory, "*.h5")))
        if not dir_sources:
            raise FileNotFoundError(f"No H5 sources found in directory: {directory}")
        srcs.extend(dir_sources)
    if not srcs:
        raise FileNotFoundError("No H5 sources provided")
    # de-duplicate while preserving order (directory listing already sorted)
    final: List[str] = []
    seen = set()
    for p in srcs:
        if p in seen:
            continue
        seen.add(p)
        final.append(p)
    return final


def _collect_video_mapping(sources: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Gather (source_idx, video_id, video_name) rows across shards."""

    src_indices: List[int] = []
    video_ids: List[int] = []
    video_names: List[str] = []

    for si, spath in enumerate(sources):
        with h5py.File(spath, "r") as f:
            if "videos/id" not in f or "videos/name" not in f:
                raise KeyError(
                    f"Shard '{spath}' is missing required videos/id or videos/name datasets"
                )
            ids = f["videos/id"][...]
            names = f["videos/name"][...]
            for raw_id, raw_name in zip(ids.tolist(), names.tolist()):
                src_indices.append(si)
                video_ids.append(int(raw_id))
                if isinstance(raw_name, (bytes, bytearray)):
                    video_names.append(raw_name.decode("utf-8"))
                else:
                    video_names.append(str(raw_name))

    if not src_indices:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), []

    return (
        np.asarray(src_indices, dtype=np.int64),
        np.asarray(video_ids, dtype=np.int64),
        video_names,
    )


def _build_video_presence(sources: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Return mapping video_name -> list of (source_idx, video_id_in_source)."""
    out: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for si, p in enumerate(sources):
        with h5py.File(p, "r") as f:
            names = f["videos/name"][...].astype(str)
            ids = f["videos/id"][...].astype(np.int32)
            for vid, nm in zip(ids.tolist(), names.tolist()):
                out[nm].append((si, int(vid)))
    return out

def _prune_broken_sources(sources: List[str]) -> List[str]:
    """Drop shards that cannot be opened with h5py, warning the user."""

    valid: List[str] = []
    skipped: List[str] = []

    for path in sources:
        print(f"Checking shard: {path}")
        try:
            with h5py.File(path, "r") as hf:
                if "frames" not in hf:
                    skipped.append(path)
                    print(
                        f"Warning: skipping shard without /frames dataset '{path}'"
                    )
                    continue
        except OSError as exc:
            skipped.append(path)
            print(f"Warning: skipping unreadable shard '{path}': {exc}")
            continue
        valid.append(path)

    if skipped:
        print(
            "Warning: {} shard(s) were skipped because they could not be opened.".format(
                len(skipped)
            )
        )

    if not valid:
        raise RuntimeError("No readable H5 sources remain after removing broken shards")

    return valid


def _map_frames_to_sources(
    sources: List[str],
    needed: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[
    Dict[str, List[Tuple[int, int]]],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """Return mapping info per split for building the VDS.

    Returns three dictionaries keyed by split name:
      - list of (source_idx, global_frame_index_in_source)
      - array of video_id values in output order
      - array of frame_idx values in output order
    """
    video_presence = _build_video_presence(sources)
    result_items: Dict[str, List[Tuple[int, int]]] = {k: [] for k in needed.keys()}
    result_vids: Dict[str, List[int]] = {k: [] for k in needed.keys()}
    result_fidx: Dict[str, List[int]] = {k: [] for k in needed.keys()}

    # Pre-open H5 files for efficiency
    open_files = [h5py.File(p, "r") for p in sources]
    try:
        for split_name, per_video in needed.items():
            for vname, frame_idxs in per_video.items():
                if vname not in video_presence:
                    continue  # video not present in any source
                # Some datasets may contain the same video in multiple shards; map greedily in order
                remaining = set(int(x) for x in frame_idxs.tolist())
                if not remaining:
                    continue
                for (si, vid) in video_presence[vname]:
                    f = open_files[si]
                    #TODO I need to store also the video name not only the id 
                    vid_arr = f["index/video_id"][...]
                    fidx_arr = f["index/frame_idx"][...]
                    pos = np.where(vid_arr == vid)[0]
                    if pos.size == 0:
                        continue
                    frames_for_vid = fidx_arr[pos]
                    # Build dict from frame_idx -> absolute index position
                    mapping = {int(fi): int(p) for fi, p in zip(frames_for_vid.tolist(), pos.tolist())}
                    take_here = np.array([fi for fi in remaining if fi in mapping], dtype=np.int64)
                    if take_here.size == 0:
                        continue
                    # Append in the order of frame_idx as they appear in the shard (mapping order)
                    # Keep original shard order by sorting by mapped position
                    mapped_positions = np.array([mapping[int(fi)] for fi in take_here], dtype=np.int64)
                    order = np.argsort(mapped_positions)
                    for idx in order:
                        result_items[split_name].append((si, int(mapped_positions[idx])))
                        result_vids[split_name].append(int(vid))
                        result_fidx[split_name].append(int(take_here[idx]))
                    remaining.difference_update(int(x) for x in take_here.tolist())
                    if not remaining:
                        break
                # Any remaining frames couldn't be found; ignored
            print(f"Mapped split '{split_name}' -> {len(result_items[split_name])} frames")
    finally:
        for f in open_files:
            f.close()
    result_vids_arr: Dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.int64) for k, v in result_vids.items()
    }
    result_fidx_arr: Dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.int64) for k, v in result_fidx.items()
    }
    return result_items, result_vids_arr, result_fidx_arr


def _coalesce_runs(items: List[Tuple[int, int]]) -> Dict[int, List[Tuple[int, int, int]]]:
    """Group (source_idx, pos) pairs into contiguous runs per source.

    Returns: source_idx -> list of (src_start, dst_start, length)
    """
    runs: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    if not items:
        return runs
    # We assume items are in desired output order; coalesce per-source consecutive positions
    dst = 0
    i = 0
    n = len(items)
    while i < n:
        si, pos = items[i]
        src_start = pos
        dst_start = dst
        length = 1
        i += 1
        dst += 1
        while i < n and items[i][0] == si and items[i][1] == (pos + length):
            length += 1
            i += 1
            dst += 1
        runs[si].append((src_start, dst_start, length))
    return runs


def _compute_contiguous_runs(
    video_ids: np.ndarray, frame_idxs: np.ndarray
) -> np.ndarray:
    """Return array [N, 4] of (dst_start, length, video_id, frame_start)."""
    if video_ids.size == 0:
        return np.empty((0, 4), dtype=np.int64)

    # Estimate the per-video stride so we can treat equally spaced frames as contiguous
    stride_by_video: Dict[int, int] = {}
    for i in range(1, video_ids.size):
        if video_ids[i] != video_ids[i - 1]:
            continue
        delta = int(frame_idxs[i]) - int(frame_idxs[i - 1])
        if delta <= 0:
            continue
        vid = int(video_ids[i])
        prev_stride = stride_by_video.get(vid)
        if prev_stride is None:
            stride_by_video[vid] = delta
        else:
            stride_by_video[vid] = int(np.gcd(prev_stride, delta))

    for vid in np.unique(video_ids.astype(np.int64)):
        iv = int(vid)
        if stride_by_video.get(iv, 0) <= 0:
            stride_by_video[iv] = 1

    segments: List[Tuple[int, int, int, int]] = []
    run_start = 0
    n = video_ids.size
    for i in range(1, n):
        same_video = video_ids[i] == video_ids[i - 1]
        delta = int(frame_idxs[i]) - int(frame_idxs[i - 1])
        if same_video and delta == stride_by_video.get(int(video_ids[i]), 1):
            continue

        run_len = i - run_start
        segments.append(
            (run_start, run_len, int(video_ids[i - 1]), int(frame_idxs[run_start]))
        )
        run_start = i

    run_len = n - run_start
    segments.append((run_start, run_len, int(video_ids[-1]), int(frame_idxs[run_start])))
    return np.asarray(segments, dtype=np.int64)


def _create_split_vds(
    out_file: h5py.File,
    group_name: str,
    sources: List[str],
    items: List[Tuple[int, int]],
    video_ids: np.ndarray,
    frame_idxs: np.ndarray,
) -> None:
    # Read shape/dtype from the first source
    source_shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {}
    with h5py.File(sources[0], "r") as f0:
        frames0: h5py.Dataset = f0["frames"]
        _, H, W, C = frames0.shape
        dtype_frames = frames0.dtype
        dtype_vid = f0["index/video_id"].dtype
        dtype_fidx = f0["index/frame_idx"].dtype
        source_shapes[sources[0]] = (
            tuple(int(x) for x in frames0.shape),
            tuple(int(x) for x in f0["index/video_id"].shape),
            tuple(int(x) for x in f0["index/frame_idx"].shape),
        )

    for spath in sources[1:]:
        with h5py.File(spath, "r") as sf:
            source_shapes[spath] = (
                tuple(int(x) for x in sf["frames"].shape),
                tuple(int(x) for x in sf["index/video_id"].shape),
                tuple(int(x) for x in sf["index/frame_idx"].shape),
            )

    N = len(items)
    grp = out_file.require_group(group_name)
    grp_index = grp.require_group("index")

    layout_frames = h5py.VirtualLayout(shape=(N, H, W, C), dtype=dtype_frames)
    layout_vid = h5py.VirtualLayout(shape=(N,), dtype=dtype_vid)
    layout_fidx = h5py.VirtualLayout(shape=(N,), dtype=dtype_fidx)

    if items:
        vds_runs = _coalesce_runs(items)
        for si, runs in vds_runs.items():
            spath = sources[si]
            frame_shape, vid_shape, fidx_shape = source_shapes[spath]
            for src_start, dst_start, length in runs:
                vsrc_frames = h5py.VirtualSource(spath, "frames", shape=frame_shape)
                vsrc_vid = h5py.VirtualSource(spath, "index/video_id", shape=vid_shape)
                vsrc_fidx = h5py.VirtualSource(spath, "index/frame_idx", shape=fidx_shape)

                dst_slice = slice(dst_start, dst_start + length)
                src_slice = slice(src_start, src_start + length)

                layout_frames[dst_slice, :, :, :] = vsrc_frames[src_slice, :, :, :]
                layout_vid[dst_slice] = vsrc_vid[src_slice]
                layout_fidx[dst_slice] = vsrc_fidx[src_slice]

    grp.create_virtual_dataset("frames", layout_frames, fillvalue=0)
    grp_index.create_virtual_dataset("video_id", layout_vid, fillvalue=-1)
    grp_index.create_virtual_dataset("frame_idx", layout_fidx, fillvalue=-1)

    if items:
        source_indices = np.asarray([si for si, _ in items], dtype=np.int64)
    else:
        source_indices = np.empty((0,), dtype=np.int64)
    grp_index.create_dataset("source_index", data=source_indices, dtype=np.int64)

    segments = _compute_contiguous_runs(video_ids, frame_idxs)
    seg_ds = grp_index.create_dataset("contiguous_runs", data=segments, dtype=np.int64)
    seg_ds.attrs["columns"] = np.array(
        ["start", "length", "video_id", "frame_start"], dtype=h5py.string_dtype("utf-8")
    )


SPLIT_NAMES: Tuple[str, ...] = ("train", "intermixed", "val", "test")


def main(args: argparse.Namespace) -> None:
    sources = _collect_sources(args.h5_sources, args.h5_dir)
    sources = _prune_broken_sources(sources)
    video_source_idx_table, video_ids_table, video_names_table = _collect_video_mapping(sources)

    subsplit_columns = tuple(
        dict.fromkeys(
            col for col in args.subsplit_cols if col and col != args.split_col
        )
    )

    needed, subsplit_needed, subsplit_meta = _read_parquet_groups(
        parquet_path=args.parquet,
        video_col=args.video_col,
        frame_idx_col=args.frame_idx_col,
        split_col=args.split_col,
        allowed_splits=SPLIT_NAMES,
        subsplit_cols=subsplit_columns,
    )

    # 2) Map requested frames to (source_idx, global index) items per split
    mapped_items, mapped_video_ids, mapped_frame_idxs = _map_frames_to_sources(sources, needed)

    subsplit_items: Dict[str, List[Tuple[int, int]]] = {}
    subsplit_video_ids: Dict[str, np.ndarray] = {}
    subsplit_frame_idxs: Dict[str, np.ndarray] = {}
    if subsplit_needed:
        (
            subsplit_items,
            subsplit_video_ids,
            subsplit_frame_idxs,
        ) = _map_frames_to_sources(sources, subsplit_needed)

    # 3) Create VDS file with groups train/intermixed/val/test
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w", libver="latest") as out:
        attrs = {
            "type": "vds_splits",
            "sources_count": len(sources),
            "sources": np.array([s.encode("utf-8") for s in sources], dtype=h5py.string_dtype("utf-8")),
            "parquet": args.parquet,
            "video_col": args.video_col,
            "frame_idx_col": args.frame_idx_col,
            "split_column": args.split_col,
            "splits": np.array(SPLIT_NAMES, dtype=h5py.string_dtype("utf-8")),
        }
        out.attrs.update(attrs)

        videos_group = out.require_group("videos")
        dt_str = h5py.string_dtype("utf-8")
        videos_group.create_dataset("source_index", data=video_source_idx_table, dtype=np.int64)
        videos_group.create_dataset("id", data=video_ids_table, dtype=np.int64)
        videos_group.create_dataset(
            "name",
            data=np.asarray(video_names_table, dtype=dt_str),
            dtype=dt_str,
        )

        for split_name in SPLIT_NAMES:
            _create_split_vds(
                out,
                split_name,
                sources,
                mapped_items.get(split_name, []),
                mapped_video_ids.get(split_name, np.empty((0,), dtype=np.int64)),
                mapped_frame_idxs.get(split_name, np.empty((0,), dtype=np.int64)),
            )
            print(f"Created split '{split_name}' with {len(mapped_items.get(split_name, []))} frames")

        if subsplit_items:
            subsplit_options: Dict[str, List[str]] = defaultdict(list)
            for group_name, items in subsplit_items.items():
                _create_split_vds(
                    out,
                    group_name,
                    sources,
                    items,
                    subsplit_video_ids.get(
                        group_name, np.empty((0,), dtype=np.int64)
                    ),
                    subsplit_frame_idxs.get(
                        group_name, np.empty((0,), dtype=np.int64)
                    ),
                )
                grp = out[group_name]
                grp.attrs["parent_split"] = group_name.split("/", 1)[0]
                for col, val in subsplit_meta.get(group_name, ()):
                    grp.attrs[col] = val
                desc = "/".join(
                    f"{col}={val}" for col, val in subsplit_meta.get(group_name, ())
                )
                subsplit_options[grp.attrs["parent_split"]].append(desc)
                print(
                    f"Created subsplit '{group_name}' with {len(items)} frames"
                )

            dt_str = h5py.string_dtype("utf-8")
            out.attrs["subsplit_columns"] = np.asarray(subsplit_columns, dtype=dt_str)
            out.attrs["subsplit_groups"] = np.asarray(
                sorted(subsplit_items.keys()), dtype=dt_str
            )
            options_grp = out.require_group("subsplit_options")
            for split_name, options in subsplit_options.items():
                data = np.asarray(sorted(set(options)), dtype=dt_str)
                options_grp.create_dataset(split_name, data=data, dtype=dt_str)

    print(f"Wrote VDS splits: {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Create VDS train/intermixed/val/test splits from shards using parquet labels"
    )
    par = p.add_argument_group("Parquet")
    par.add_argument("--parquet", type=str, required=True, help="Path to labels parquet")
    par.add_argument("--video_col", type=str, default="video_id", help="Column: video basename without extension")
    par.add_argument("--frame_idx_col", type=str, default="idx", help="Column: integer frame index in source video")
    par.add_argument("--split_col", type=str, default="split", help="Column whose values define dataset splits")
    par.add_argument(
        "--subsplit-cols",
        nargs="*",
        default=(),
        help="Optional additional categorical columns to nest under each split",
    )

    p.add_argument("--h5-sources", dest="h5_sources", nargs='*', default=[], help="Explicit shard H5 paths")
    p.add_argument("--h5-dir", dest="h5_dir", type=str, default="", help="Directory containing shard H5 files")

    out = p.add_argument_group("Output")
    out.add_argument("--output", type=str, required=True, help="Output VDS H5 path")

    args = p.parse_args()
    main(args)
