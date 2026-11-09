"""Train one arm of Test 4 with lerobot's own training loop (lerobot 0.6.1 lerobot-train).

    python train_wam.py --arm B --output_dir runs/armB --targets_dir T --dataset_root D \
        --policy_path CKPT [--steps 6000 --batch_size 16 --future_weight 0.1 --save_freq 1000]
    python train_wam.py --arm B --output_dir runs/armB --targets_dir T --resume

How: we call ``lerobot.scripts.lerobot_train.train()`` unchanged, after replacing four names in
that module (the installed packages are not edited):

  make_policy               -> stock policy, then attach_wam() (arm A: logging only; B/C: head)
  make_pre_post_processors  -> stock pipelines; the preprocessor is wrapped by CarryIndex so the
                               batch keeps episode_index / frame_index for the target lookup
  save_checkpoint           -> stock save (head stripped from model.safetensors), plus
                               wam_head.safetensors + wam_config.json next to pretrained_model/
  update_policy             -> stock step, plus peak-GPU-memory / tokens-per-sample printing

So arm A is lerobot-train with the Griffin recipe (batch 16, 6000 steps, seed 1000) and nothing
else; its loss is the stock flow loss tensor. Any argument we do not know is passed to lerobot
unchanged (e.g. --policy.require_attn_implementation=true).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import wam_head as W  # noqa: E402

RECIPE = {  # docs/finetuning.md and README.md of alpha-s @ dbbbf9a
    "policy_path": "griffinlabs/Griffin-Alpha-S-LIBERO",
    "dataset_repo_id": "HuggingFaceVLA/libero",
    "batch_size": 16,
    "steps": 6000,
}


def parse_args(argv):
    p = argparse.ArgumentParser(description="Test 4: Griffin Alpha-S + future head")
    p.add_argument("--arm", choices=W.ARMS, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=RECIPE["steps"])
    p.add_argument("--batch_size", type=int, default=RECIPE["batch_size"])
    p.add_argument("--future_weight", type=float, default=0.1)
    p.add_argument("--save_freq", type=int, default=1000, help="checkpoint every N steps")
    p.add_argument("--keep_last", type=int, default=2,
                   help="delete older step checkpoints (~26 GB each), keep the newest N; 0=all")
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--resume", action="store_true", help="continue from output_dir/.../last")
    p.add_argument("--weights_only", action="store_true",
                   help="save model weights only, no optimizer/scheduler state (~10 GB "
                        "instead of ~26 GB); such a checkpoint cannot be resumed")
    p.add_argument("--policy_path", default=RECIPE["policy_path"])
    p.add_argument("--dataset_repo_id", default=RECIPE["dataset_repo_id"])
    p.add_argument("--dataset_root", default=None, help="local copy of the dataset")
    p.add_argument("--targets_dir", default=None, help="dir with vjepa_targets.npy (arms B/C)")
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--head_hidden", type=int, default=2048)
    p.add_argument("--hook_layer", type=int, default=-2,
                   help="decoder layer whose OUTPUT the head reads (-2 = input of the last)")
    p.add_argument("--seed", type=int, default=1000, help="lerobot's default seed")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args, passthrough = p.parse_known_args(argv)
    if args.arm != "A" and not args.targets_dir:
        p.error(f"--arm {args.arm} needs --targets_dir")
    if args.resume and args.weights_only:
        p.error("--resume needs optimizer state; it cannot be combined with --weights_only")
    return args, passthrough


def lerobot_argv(args, passthrough) -> list[str]:
    out = Path(args.output_dir)
    if args.resume:
        cfg = out / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"nothing to resume: {cfg} does not exist")
        argv = [f"--config_path={cfg}", "--resume=true", f"--steps={args.steps}"]
    else:
        argv = [
            f"--policy.path={args.policy_path}",
            f"--dataset.repo_id={args.dataset_repo_id}",
            f"--policy.device={args.device}",
            "--policy.push_to_hub=false",
            f"--batch_size={args.batch_size}",
            f"--steps={args.steps}",
            f"--output_dir={out}",
            f"--save_freq={args.save_freq}",
            f"--log_freq={args.log_freq}",
            f"--seed={args.seed}",
            f"--num_workers={args.num_workers}",
        ]
        if args.dataset_root:
            argv.append(f"--dataset.root={args.dataset_root}")
    return argv + list(passthrough)


class Monitor:
    """Peak GPU memory and prompt length, printed as grep-able WAM_STATS lines."""

    def __init__(self, log_freq: int):
        self.log_freq = log_freq
        self.steps = 0
        self.peak_gb = 0.0
        self.tok = []
        self.img_tok = []
        self.t0 = time.time()

    def after_step(self, output_dict) -> None:
        self.steps += 1
        if torch.cuda.is_available():
            self.peak_gb = max(self.peak_gb, torch.cuda.max_memory_allocated() / 2**30)
        if output_dict:
            if "tok_per_sample" in output_dict:
                self.tok.append(output_dict["tok_per_sample"])
            if "img_tok_main" in output_dict:
                self.img_tok.append(output_dict["img_tok_main"])
        if self.steps in (1, 2, 10) or self.steps % self.log_freq == 0:
            print("WAM_STATS " + json.dumps(self.summary()), flush=True)

    def summary(self) -> dict:
        n = max(len(self.tok), 1)
        return {
            "steps_this_process": self.steps,
            "peak_gpu_mem_gb": round(self.peak_gb, 2),
            "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
            if torch.cuda.is_available() else None,
            "tokens_per_sample_mean": round(sum(self.tok) / n, 1),
            "main_image_tokens_mean": round(sum(self.img_tok) / max(len(self.img_tok), 1), 1),
            "sec_per_step": round((time.time() - self.t0) / max(self.steps, 1), 3),
        }


def prune(checkpoints_dir: Path, keep_last: int) -> None:
    """Remove all but the newest ``keep_last`` step dirs (all-digit names) in checkpoints/."""
    import shutil

    if keep_last <= 0:
        return
    steps = sorted(d for d in checkpoints_dir.iterdir()
                   if d.is_dir() and not d.is_symlink() and d.name.isdigit())
    for old in steps[:-keep_last]:
        logging.info("WAM prune: removing %s", old)
        shutil.rmtree(old)


def install(args, lt) -> Monitor:
    """Patch the four names in the lerobot_train module ``lt``. Returns the monitor."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    wcfg = W.WAMConfig(arm=args.arm, future_weight=args.future_weight,
                       head_hidden=args.head_hidden, head_lr=args.head_lr,
                       hook_layer=args.hook_layer)
    monitor = Monitor(args.log_freq)
    orig_make_policy = lt.make_policy
    orig_make_pp = lt.make_pre_post_processors
    orig_save = lt.save_checkpoint
    orig_update = lt.update_policy

    def make_policy(*a, **k):
        policy = orig_make_policy(*a, **k)
        cfg = k.get("cfg", a[0] if a else None)
        targets, head_state = None, None
        if wcfg.target_kind is not None:
            targets = W.TargetTable.load(args.targets_dir, horizon=wcfg.horizon)
            logging.info("WAM targets: %d rows, %d valid for t+%d",
                         targets.emb.shape[0], targets.stats["n_valid"], wcfg.horizon)
        if args.resume:
            ckpt = Path(cfg.pretrained_path).parent
            saved = json.loads((ckpt / W.WAM_CONFIG_FILE).read_text())
            for key in ("arm", "future_weight", "head_hidden", "hook_layer"):
                if saved.get(key, -2 if key == "hook_layer" else None) != getattr(wcfg, key):
                    raise ValueError(f"resume: {key}={getattr(wcfg, key)} but checkpoint has "
                                     f"{saved[key]}")
            if wcfg.target_kind is not None:
                head_state = load_file(str(ckpt / W.HEAD_FILE))
                logging.info("WAM resume: head loaded from %s", ckpt / W.HEAD_FILE)
        W.attach_wam(policy, wcfg, targets, head_state)
        if hasattr(policy, "wam_head"):
            n = sum(p.numel() for p in policy.wam_head.parameters())
            logging.info("WAM arm %s: head %.2fM params, weight %g, head_lr %g, hook layers[%d]",
                         wcfg.arm, n / 1e6, wcfg.future_weight, wcfg.head_lr, wcfg.hook_layer)
        else:
            logging.info("WAM arm A: stock flow loss, no head")
        return policy

    def make_pre_post_processors(*a, **k):
        pre, post = orig_make_pp(*a, **k)
        return W.CarryIndex(pre), post

    def save_checkpoint(*a, **k):
        if args.weights_only:  # lerobot's save_training_state accepts None for both
            k = {**k, "optimizer": None, "scheduler": None, "optim_state_dict": None}
        orig_save(*a, **k)
        ckpt = Path(k["checkpoint_dir"])
        policy = k["policy"]
        weights = ckpt / "pretrained_model" / "model.safetensors"
        with safe_open(str(weights), framework="pt") as f:
            bad = [key for key in f.keys() if key.startswith(W.HEAD_PREFIX)]
        if bad:
            raise RuntimeError(f"head keys leaked into {weights}: {bad[:3]}")
        if hasattr(policy, "wam_head"):
            save_file(W.head_state_dict(policy), str(ckpt / W.HEAD_FILE))
        wcfg.save(str(ckpt))
        prune(ckpt.parent, args.keep_last)
        print("WAM_STATS " + json.dumps({"checkpoint": str(ckpt), **monitor.summary()}),
              flush=True)

    def update_policy(*a, **k):
        tracker, output_dict = orig_update(*a, **k)
        monitor.after_step(output_dict)
        return tracker, output_dict

    lt.make_policy = make_policy
    lt.make_pre_post_processors = make_pre_post_processors
    lt.save_checkpoint = save_checkpoint
    lt.update_policy = update_policy
    return monitor


def main(argv=None) -> None:
    args, passthrough = parse_args(sys.argv[1:] if argv is None else argv)
    import lerobot_policy_griffin_alpha  # noqa: F401  (registers griffin_alpha)
    from lerobot.scripts import lerobot_train as lt
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    monitor = install(args, lt)
    sys.argv = [sys.argv[0]] + lerobot_argv(args, passthrough)
    print("WAM lerobot argv: " + " ".join(sys.argv[1:]), flush=True)
    lt.train()
    print("WAM_FINAL " + json.dumps(monitor.summary()), flush=True)


if __name__ == "__main__":
    main()
