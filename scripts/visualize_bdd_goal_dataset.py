#!/usr/bin/env python3
"""
Utility to sample BDDGoalDataset clips and visualize their context/goal locations on an interactive map.
Each sampled clip adds start/context/goal markers and an inline frame image so we can inspect alignment.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Iterable, List

import folium
import numpy as np
import torch
from PIL import Image

from data.Goal_BDD import BDDGoalDataset

EARTH_RADIUS_M = 6_371_000.0


def _frame_to_png_bytes(frame_tensor: torch.Tensor) -> bytes:
    frame = frame_tensor.detach().cpu()
    frame = ((frame + 1.0) / 2.0).clamp(0.0, 1.0)
    frame_uint8 = (frame * 255.0).byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(frame_uint8)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _offset_latlon(lat: float, lon: float, east_m: float, north_m: float) -> List[float]:
    dlat = (north_m / EARTH_RADIUS_M) * (180.0 / np.pi)
    if abs(lat) >= 90:
        dlon = 0.0
    else:
        dlon = (east_m / (EARTH_RADIUS_M * np.cos(np.radians(lat)))) * (180.0 / np.pi)
    return [lat + dlat, lon + dlon]


def _add_sample_markers(
    fmap: folium.Map,
    sample_idx: int,
    sample: dict,
    ctx_png: bytes,
    goal_png: bytes,
) -> None:
    start_latlon = sample["start_latlon"].cpu().numpy().tolist()
    ctx_latlon = sample["context_latlon"].cpu().numpy().tolist()
    goal_latlon = sample["goal_latlon"].cpu().numpy().tolist()
    relative_goal = sample["relative_goal_xy"].cpu().numpy()
    context_xy = sample["context_position_xy"].cpu().numpy()
    goal_xy = sample["goal_position_xy"].cpu().numpy()
    direction = sample["direction"].cpu().numpy()
    yaw = float(sample["yaw_rad"])
    heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    right = np.array([heading[1], -heading[0]], dtype=np.float32)
    rotation = np.stack([right, heading], axis=0)
    direction_global = rotation.T @ direction
    arrow_len_m = max(float(sample["speed_ms"]) * 2.0, 5.0)
    heading_tip = _offset_latlon(
        ctx_latlon[0],
        ctx_latlon[1],
        east_m=direction_global[0] * arrow_len_m,
        north_m=direction_global[1] * arrow_len_m,
    )

    popup_html = f"""
    <div style="width:260px">
        <h4>{sample['video_id']} · idx {sample_idx}</h4>
        <b>Context</b><br/>
        <img src="data:image/png;base64,{base64.b64encode(ctx_png).decode('ascii')}" width="240"/><br/>
        <b>Goal</b><br/>
        <img src="data:image/png;base64,{base64.b64encode(goal_png).decode('ascii')}" width="240"/><br/>
        <b>Time of day:</b> {sample['time_of_day_label']}<br/>
        <b>Speed:</b> {float(sample['speed_ms']):.2f} m/s<br/>
        <b>Yaw:</b> {float(sample['yaw_rad']) * 180.0 / np.pi:.1f}°<br/>
        <b>Ctx XY:</b> ({context_xy[0]:+.1f}, {context_xy[1]:+.1f}) m<br/>
        <b>Goal Δ:</b> ({relative_goal[0]:+.1f}, {relative_goal[1]:+.1f}) m
    </div>
    """
    iframe = folium.IFrame(html=popup_html, width=260, height=260)
    folium.Marker(
        location=ctx_latlon,
        popup=folium.Popup(iframe),
        icon=folium.Icon(color="blue", icon="camera"),
    ).add_to(fmap)

    folium.Marker(
        location=goal_latlon,
        popup=folium.Popup(f"<img src='data:image/png;base64,{base64.b64encode(goal_png).decode('ascii')}' width='240'/>"),
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(fmap)

    folium.CircleMarker(location=start_latlon, radius=5, color="green", fill=True, fill_opacity=0.8, tooltip="clip start").add_to(fmap)
    goal_distance_m = float(np.linalg.norm(goal_xy))
    folium.PolyLine(
        locations=[start_latlon, goal_latlon],
        color="orange",
        weight=2,
        opacity=0.8,
        tooltip=f"distance: {goal_distance_m:.1f} m",
    ).add_to(fmap)
    folium.PolyLine(locations=[ctx_latlon, heading_tip], color="blue", weight=3, opacity=0.85, tooltip="heading").add_to(fmap)


def _sample_indices(length: int, count: int, seed: int) -> Iterable[int]:
    rng = np.random.default_rng(seed)
    count = min(count, length)
    if count <= 0:
        return []
    return rng.choice(length, size=count, replace=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mp4-list", type=Path, required=True, help="Text file with absolute MP4 paths, one per line.")
    parser.add_argument("--info-root", type=Path, required=True, help="Directory containing BDD100K info JSON files.")
    parser.add_argument("--cache-dir", type=Path, default=Path("metadata_cache"), help="Directory to store per-video metadata caches.")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames per clip.")
    parser.add_argument("--ctx-frame", type=int, default=0, help="Which frame (0-indexed) to treat as the context/current timestep.")
    parser.add_argument("--direction-window", type=int, default=5, help="Window length (frames) for direction estimation.")
    parser.add_argument("--stored-fps", type=int, default=10, help="Recorded FPS of the MP4 files.")
    parser.add_argument("--frame-rate", type=int, default=5, help="Target FPS for sampling clips.")
    parser.add_argument("--samples", type=int, default=8, help="Number of dataset items to visualize.")
    parser.add_argument("--output", type=Path, default=Path("bdd_goal_samples.html"), help="Output HTML map path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling dataset entries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = BDDGoalDataset(
        mp4_paths=[args.mp4_list],
        info_root=args.info_root,
        metadata_cache_dir=args.cache_dir,
        num_frames=args.num_frames,
        ctx_frame=args.ctx_frame,
        direction_window=args.direction_window,
        stored_data_frame_rate=args.stored_fps,
        frame_rate=args.frame_rate,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; check MP4 list and parameters.")

    indices = list(_sample_indices(len(dataset), args.samples, args.seed))
    samples = [dataset[i] for i in indices]
    center_latlon = samples[0]["context_latlon"].cpu().numpy().tolist()
    fmap = folium.Map(location=center_latlon, zoom_start=14, tiles="cartodbpositron")

    for ds_idx, sample in zip(indices, samples):
        ctx_png = _frame_to_png_bytes(sample["images"][dataset.ctx_frame])
        goal_png = _frame_to_png_bytes(sample["images"][-1])
        _add_sample_markers(fmap, ds_idx, sample, ctx_png, goal_png)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(args.output))
    print(f"Wrote map with {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
