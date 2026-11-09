"""Precompute frozen V-JEPA 2 targets: one 1,024-dim fp16 vector per LIBERO frame.

For every frame t of every episode we feed V-JEPA 2 a short clip that ENDS at t:
frames t-(T-1)*s, ..., t-s, t (T = --clip_len, s = --stride; indices before 0 repeat frame 0),
take the encoder's last_hidden_state (no predictor) and mean-pool its tokens to one vector.

Defaults: T=8, s=2 -> the clip spans 1.4 s of the 10 fps video (an effective 5 fps, close to
the low frame rates V-JEPA 2 was trained on); 8 frames / tubelet 2 = 4 time slots x 256 patches
= 1,024 tokens. Cost is ~0.7 TFLOP per clip, ~2e17 FLOP for 273,465 frames: well under an hour
on one H100/A100 in bf16. Preprocessing follows VJEPA2VideoProcessor (resize shortest edge to
crop*256/224 = 292, centre-crop 256, ImageNet mean/std).

Input: the HuggingFaceVLA/libero snapshot (LeRobot v3; images are PNG bytes inside
data/chunk-*/file-*.parquet, column observation.images.image = main/agent-view camera).
Output (in --out_dir):
  parts/ep_00000.npy ...   one (L, 1024) fp16 array per episode (atomic writes => resumable)
  vjepa_targets.npy        (N, 1024) fp16, rows grouped by episode, frames 0..L-1   [--merge]
  vjepa_index.npz          episode_index, frame_index (N,) + the settings used      [--merge]

Usage:
  python vjepa_targets.py --data_root D --out_dir O [--episodes 0 847]   # GPU job(s)
  python vjepa_targets.py --data_root D --out_dir O --merge              # CPU, after all parts
  python vjepa_targets.py --data_root D --out_dir O --tiny_random        # CPU smoke test
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

IMAGE_KEY = "observation.images.image"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------------------------
# Dataset scan
# ------------------------------------------------------------------------------------------
def parquet_files(data_root: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_root, "data", "chunk-*", "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no data/chunk-*/file-*.parquet under {data_root}")
    return files


def scan(data_root: str):
    """-> (files, per-file episode arrays, episode -> length). Reads two int columns only."""
    import pyarrow.parquet as pq

    files = parquet_files(data_root)
    per_file = []
    lengths: dict[int, int] = {}
    for path in files:
        t = pq.read_table(path, columns=["episode_index", "frame_index"])
        ep = t["episode_index"].to_numpy()
        per_file.append(np.unique(ep))
        for e, n in zip(*np.unique(ep, return_counts=True)):
            lengths[int(e)] = lengths.get(int(e), 0) + int(n)
    return files, per_file, lengths


# ------------------------------------------------------------------------------------------
# Clips and preprocessing
# ------------------------------------------------------------------------------------------
def clip_indices(length: int, clip_len: int, stride: int) -> torch.Tensor:
    """(length, clip_len) frame indices; row t ends at t, earlier indices clamp to 0."""
    t = torch.arange(length)[:, None]
    back = (clip_len - 1 - torch.arange(clip_len))[None, :] * stride
    return (t - back).clamp_min(0)


def preprocess(frames_u8: torch.Tensor, crop: int = 256) -> torch.Tensor:
    """(L, H, W, 3) uint8 -> (L, 3, crop, crop) float, VJEPA2VideoProcessor-style."""
    x = frames_u8.permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    short = int(crop * 256 / 224)
    scale = short / min(h, w)
    size = (int(round(h * scale)), int(round(w * scale)))
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    top = (size[0] - crop) // 2
    left = (size[1] - crop) // 2
    x = x[:, :, top:top + crop, left:left + crop]
    mean = torch.tensor(IMAGENET_MEAN, device=x.device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, device=x.device)[None, :, None, None]
    return (x - mean) / std


def pool_tokens(tokens: torch.Tensor, pool: str, tokens_per_slot: int) -> torch.Tensor:
    """(B, N, D) -> (B, D). 'all' = mean of every token; 'last' = last time slot only."""
    if pool == "all":
        return tokens.float().mean(dim=1)
    if pool == "last":
        return tokens[:, -tokens_per_slot:].float().mean(dim=1)
    raise ValueError(pool)


@torch.no_grad()
def embed_episode(model, frames_u8, clip_len, stride, pool, batch_clips, device, dtype):
    frames = preprocess(frames_u8.to(device)).to(dtype)  # (L, 3, 256, 256)
    idx = clip_indices(frames.shape[0], clip_len, stride).to(device)
    grid = model.config.crop_size // model.config.patch_size
    out = []
    for s in range(0, idx.shape[0], batch_clips):
        clips = frames[idx[s:s + batch_clips]]  # (b, T, 3, 256, 256)
        tokens = model(pixel_values_videos=clips, skip_predictor=True).last_hidden_state
        out.append(pool_tokens(tokens, pool, grid * grid).cpu())
    return torch.cat(out).to(torch.float16).numpy()


# ------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------
def load_model(name: str, tiny_random: bool, device, dtype):
    from transformers import VJEPA2Config, VJEPA2Model

    if tiny_random:
        cfg = VJEPA2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                           pred_hidden_size=32, pred_num_hidden_layers=1,
                           pred_num_attention_heads=4, crop_size=256, frames_per_clip=8)
        model = VJEPA2Model(cfg)
    else:
        model = VJEPA2Model.from_pretrained(name, dtype=dtype)
    return model.to(device=device, dtype=dtype).eval()


# ------------------------------------------------------------------------------------------
# Main loops
# ------------------------------------------------------------------------------------------
def part_path(out_dir: str, ep: int) -> str:
    return os.path.join(out_dir, "parts", f"ep_{ep:05d}.npy")


def save_atomic(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def decode_png(blob: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))


def run(args) -> None:
    import pyarrow.parquet as pq

    os.makedirs(os.path.join(args.out_dir, "parts"), exist_ok=True)
    files, per_file, lengths = scan(args.data_root)
    lo, hi = args.episodes if args.episodes else (0, max(lengths) + 1)
    todo = {e for e in lengths if lo <= e < hi and not os.path.exists(part_path(args.out_dir, e))}
    if args.max_episodes:
        todo = set(sorted(todo)[: args.max_episodes])
    print(f"episodes in range [{lo},{hi}): {sum(lo <= e < hi for e in lengths)}, "
          f"to do: {len(todo)}", flush=True)
    if not todo:
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = load_model(args.model, args.tiny_random, device, dtype)
    pool = ThreadPoolExecutor(max_workers=args.decode_threads)
    buf: dict[int, dict[int, np.ndarray]] = {}
    done, n_frames, t0 = 0, 0, time.time()
    for path, eps in zip(files, per_file):
        want = [int(e) for e in eps if int(e) in todo]
        if not want:
            continue
        t = pq.read_table(path, columns=[IMAGE_KEY, "episode_index", "frame_index"])
        ep_col = t["episode_index"].to_numpy()
        fr_col = t["frame_index"].to_numpy()
        rows = np.nonzero(np.isin(ep_col, want))[0]
        blobs = t.take(rows)[IMAGE_KEY].combine_chunks().field("bytes").to_pylist()
        images = list(pool.map(decode_png, blobs))
        for r, img in zip(rows, images):
            buf.setdefault(int(ep_col[r]), {})[int(fr_col[r])] = img
        for e in want:
            got = buf.get(e, {})
            if len(got) < lengths[e]:
                continue  # episode continues in the next file
            if sorted(got) != list(range(lengths[e])):
                raise ValueError(f"episode {e}: frame_index is not 0..{lengths[e] - 1}")
            frames = torch.from_numpy(np.stack([got[f] for f in range(lengths[e])]))
            emb = embed_episode(model, frames, args.clip_len, args.stride, args.pool,
                                args.batch_clips, device, dtype)
            save_atomic(part_path(args.out_dir, e), emb)
            del buf[e]
            done += 1
            n_frames += lengths[e]
            if done % 20 == 0 or done == len(todo):
                rate = n_frames / max(time.time() - t0, 1e-6)
                print(f"[{done}/{len(todo)}] episodes, {n_frames} frames, "
                      f"{rate:.0f} frames/s", flush=True)
    if buf:
        raise RuntimeError(f"incomplete episodes left in buffer: {sorted(buf)[:10]}")


def merge(args) -> None:
    _, _, lengths = scan(args.data_root)
    eps = sorted(lengths)
    if eps != list(range(len(eps))):
        raise ValueError("episode indices are not 0..E-1")
    missing = [e for e in eps if not os.path.exists(part_path(args.out_dir, e))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} episode parts missing, first: {missing[:10]}")
    total = sum(lengths.values())
    out = np.lib.format.open_memmap(os.path.join(args.out_dir, "vjepa_targets.npy.tmp.npy"),
                                    mode="w+", dtype=np.float16, shape=(total, args.dim))
    ep_idx = np.empty(total, dtype=np.int64)
    fr_idx = np.empty(total, dtype=np.int64)
    row = 0
    for e in eps:
        part = np.load(part_path(args.out_dir, e))
        if part.shape != (lengths[e], args.dim):
            raise ValueError(f"episode {e}: part {part.shape} != ({lengths[e]}, {args.dim})")
        if not np.isfinite(part.astype(np.float32)).all():
            raise ValueError(f"episode {e}: non-finite values")
        out[row:row + lengths[e]] = part
        ep_idx[row:row + lengths[e]] = e
        fr_idx[row:row + lengths[e]] = np.arange(lengths[e])
        row += lengths[e]
    out.flush()
    del out
    os.replace(os.path.join(args.out_dir, "vjepa_targets.npy.tmp.npy"),
               os.path.join(args.out_dir, "vjepa_targets.npy"))
    meta = {"model": args.model, "clip_len": args.clip_len, "stride": args.stride,
            "pool": args.pool, "image_key": IMAGE_KEY, "n_frames": total, "n_episodes": len(eps)}
    np.savez(os.path.join(args.out_dir, "vjepa_index.npz"), episode_index=ep_idx,
             frame_index=fr_idx, meta=json.dumps(meta))
    print(json.dumps(meta), flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_root", required=True, help="local HuggingFaceVLA/libero snapshot")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256",
                   help="HF id or local dir of the V-JEPA 2 weights")
    p.add_argument("--episodes", type=int, nargs=2, metavar=("START", "END"),
                   help="half-open episode range for this job (default: all)")
    p.add_argument("--clip_len", type=int, default=8)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--pool", choices=("all", "last"), default="all")
    p.add_argument("--batch_clips", type=int, default=32)
    p.add_argument("--decode_threads", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--max_episodes", type=int, default=0, help="debug: stop after N episodes")
    p.add_argument("--tiny_random", action="store_true",
                   help="random tiny V-JEPA (no download) for smoke tests; dim becomes 64")
    p.add_argument("--merge", action="store_true", help="assemble parts into the final files")
    args = p.parse_args(argv)
    if args.tiny_random:
        args.dim = 64
    if args.merge:
        merge(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
