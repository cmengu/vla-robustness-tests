"""Linear probe: did the "future lesson" reach the backbone?

Freeze a checkpoint, run the backbone on held-out frames, mean-pool the hidden state over the
main camera's image tokens, and fit a ridge regression to the standardised V-JEPA CHANGE target
emb(t+10) - emb(t) (and, as a reference, the CURRENT target emb(t)). Report held-out R^2 and
cosine. Run it on the A, B and C checkpoints (and the untouched HF checkpoint) with the SAME
--seed: the episode split and the sampled frames depend only on the seed and the target table,
so every checkpoint is probed on identical frames.

    python probe.py --checkpoint runs/armB/checkpoints/last/pretrained_model \
        --targets_dir T --dataset_root D --out probe_B.json [--layer -2]

--layer -2 (default) probes the output of decoder layer -2 (layer 34 of 36) = the INPUT of the
last decoder layer: the same tensor the head was trained on, and the one the last layer's K/V
(read by the action expert) are computed from. Other choices: any layer index, e.g. --layer 17
to see how deep the lesson went, or --layer final for the post-norm text-model output (which
the expert never reads). Features get the same parameter-free LayerNorm as the head input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wam_head as W  # noqa: E402


# ------------------------------------------------------------------------------------------
# Frame selection (deterministic, checkpoint-independent)
# ------------------------------------------------------------------------------------------
def select_frames(table: W.TargetTable, n_train: int, n_test: int, test_frac: float, seed: int):
    """-> dict split -> (episode (n,), frame (n,)). Split by EPISODE; only frames with t+10."""
    rng = np.random.default_rng(seed)
    n_ep = table.ep_len.shape[0]
    perm = rng.permutation(n_ep)
    n_test_ep = max(1, int(round(test_frac * n_ep)))
    split_eps = {"test": np.sort(perm[:n_test_ep]), "train": np.sort(perm[n_test_ep:])}
    lens = table.ep_len.cpu().numpy()
    out = {}
    for split, n in (("train", n_train), ("test", n_test)):
        eps = split_eps[split]
        usable = np.maximum(lens[eps] - table.horizon, 0)
        cum = np.cumsum(usable)
        if cum[-1] == 0:
            raise ValueError(f"no usable frames in {split}")
        pick = np.sort(rng.choice(cum[-1], size=min(n, int(cum[-1])), replace=False))
        which = np.searchsorted(cum, pick, side="right")
        start = np.concatenate([[0], cum[:-1]])
        out[split] = (eps[which], pick - start[which])
    return out


# ------------------------------------------------------------------------------------------
# Ridge probe
# ------------------------------------------------------------------------------------------
def r2_scores(y: np.ndarray, pred: np.ndarray, ref_mean: np.ndarray) -> dict:
    sse = ((y - pred) ** 2).sum(0)
    sst = ((y - ref_mean) ** 2).sum(0)
    per_dim = 1.0 - sse / np.maximum(sst, 1e-12)
    cos = (y * pred).sum(1) / (np.linalg.norm(y, axis=1) * np.linalg.norm(pred, axis=1) + 1e-12)
    return {"r2_mean_over_dims": float(per_dim.mean()),
            "r2_pooled": float(1.0 - sse.sum() / max(sst.sum(), 1e-12)),
            "cosine_mean": float(cos.mean())}


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float):
    xm, xs = x.mean(0), x.std(0) + 1e-6
    ym = y.mean(0)
    xz = (x - xm) / xs
    a = xz.T @ xz + lam * len(xz) * np.eye(xz.shape[1])
    w = np.linalg.solve(a, xz.T @ (y - ym))
    return lambda q: ((q - xm) / xs) @ w + ym


def ridge_probe(x_tr, y_tr, x_te, y_te, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), seed=0):
    """Pick lambda on a 20% validation slice of the probe-train frames, refit, score on test."""
    x_tr, y_tr, x_te, y_te = (a.astype(np.float64) for a in (x_tr, y_tr, x_te, y_te))
    idx = np.random.default_rng(seed).permutation(len(x_tr))
    n_val = max(1, len(idx) // 5)
    val, fit = idx[:n_val], idx[n_val:]
    best = None
    for lam in lams:
        pred = fit_ridge(x_tr[fit], y_tr[fit], lam)(x_tr[val])
        score = r2_scores(y_tr[val], pred, y_tr[fit].mean(0))["r2_pooled"]
        if best is None or score > best[1]:
            best = (lam, score)
    model = fit_ridge(x_tr, y_tr, best[0])
    res = r2_scores(y_te, model(x_te), y_tr.mean(0))
    res.update({"lambda": best[0], "val_r2_pooled": best[1]})
    return res


# ------------------------------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------------------------------
def extract(args, frames: dict, table: W.TargetTable) -> dict:
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    from lerobot_policy_griffin_alpha import GriffinAlphaPolicy

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = GriffinAlphaPolicy.from_pretrained(args.checkpoint).to(device).eval()
    pre, _ = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": device.type}})
    needed = sorted({int(e) for eps, _ in frames.values() for e in eps})
    ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=needed)
    # (episode, frame) -> row of this (episode-filtered) dataset, from its own columns
    col_ep = ds.hf_dataset.data.column("episode_index").to_numpy().astype(np.int64)
    col_fr = ds.hf_dataset.data.column("frame_index").to_numpy().astype(np.int64)
    keys = col_ep * 1_000_000 + col_fr
    order = np.argsort(keys)
    module = W.hooked_module(policy, args.layer)
    image_token_id = W.image_token_id_of(policy)
    feats = {}
    for split, (eps, frs) in frames.items():
        want = eps.astype(np.int64) * 1_000_000 + frs.astype(np.int64)
        pos = np.searchsorted(keys, want, sorter=order)
        rows = order[np.minimum(pos, len(order) - 1)]
        if not np.array_equal(keys[rows], want):
            raise ValueError("some probe frames are missing from the dataset")
        out = []
        for s in range(0, len(rows), args.batch_size):
            items = [ds[int(r)] for r in rows[s:s + args.batch_size]]
            for it, e, f in zip(items, eps[s:], frs[s:]):
                if int(it["episode_index"]) != int(e) or int(it["frame_index"]) != int(f):
                    raise ValueError("dataset row order differs from the target table")
            batch = torch.utils.data.default_collate(items)
            for key in ds.meta.camera_keys:
                if key in batch and batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0
            batch.pop("action", None)  # inference-form prompt; no augmentation under no_grad
            with torch.no_grad():
                inputs = policy._filter_model_inputs(pre(batch))
                with W.HiddenCapture(module) as cap:
                    policy._run_prefix(inputs, with_grad=False)
                if cap.calls != 1:
                    raise RuntimeError(f"hook fired {cap.calls} times")
                pooled, _ = W.pool_main_camera(cap.value, inputs["input_ids"], image_token_id)
            pooled = torch.nn.functional.layer_norm(pooled, (pooled.shape[-1],))
            out.append(pooled.cpu().numpy())
            if (s // args.batch_size) % 20 == 0:
                print(f"{split}: {s + len(items)}/{len(rows)}", flush=True)
        feats[split] = np.concatenate(out)
    return feats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="ridge probe of pooled backbone features")
    p.add_argument("--checkpoint", required=True, help=".../pretrained_model dir or HF id")
    p.add_argument("--targets_dir", required=True)
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--dataset_repo_id", default="HuggingFaceVLA/libero")
    p.add_argument("--out", required=True, help="results JSON; features cached next to it")
    p.add_argument("--layer", default="-2",
                   help="decoder layer k (output of layers[k]; -2 = training hook) or 'final'")
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--test_frac", type=float, default=0.2, help="fraction of EPISODES held out")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if args.layer != "final":
        args.layer = int(args.layer)

    t0 = time.time()
    table = W.TargetTable.load(args.targets_dir)
    frames = select_frames(table, args.n_train, args.n_test, args.test_frac, args.seed)
    cache = os.path.splitext(args.out)[0] + "_feats.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        feats = {"train": z["train"], "test": z["test"]}
        print(f"using cached features {cache}")
    else:
        feats = extract(args, frames, table)
        np.savez(cache, **feats)
    results = {"checkpoint": args.checkpoint, "layer": args.layer, "seed": args.seed,
               "n_train": len(feats["train"]), "n_test": len(feats["test"])}
    for kind in ("change", "current"):
        ys = {}
        for split, (eps, frs) in frames.items():
            y, valid = table.lookup(torch.as_tensor(eps), torch.as_tensor(frs), kind)
            assert bool(valid.all())
            ys[split] = y.numpy()
        results[kind] = ridge_probe(feats["train"], ys["train"], feats["test"], ys["test"],
                                    seed=args.seed)
    results["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
