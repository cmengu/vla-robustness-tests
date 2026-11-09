#!/usr/bin/env python
"""Test 2 (moved object): eval_delay.py plus a mid-episode sideways shift of the target object.

The rollout loop, the delay logic and the --griffin.* flags are copied unchanged from
test1/eval_delay.py (only long lines are re-wrapped to fit 100 characters). New here:

  --shift.mode=none|step0|midreach|blind  default none (no shift; same actions as eval_delay)
      step0     shift right after reset, before the first action
      midreach  shift the first step the gripper is within --shift.radius of the target
      blind     as midreach, but from then on the policy keeps getting the camera images
                captured just before the shift (robot state stays live)
  --shift.size=METRES     sideways distance, default 0.05
  --shift.radius=METRES   mid-reach trigger distance (3D, gripper site to object body), 0.15
  --shift.seed=INT        direction seed, default 0. The direction is drawn from
                          rng(seed, task_id, start_state), so every mode and model gets the
                          same direction sequence for the same episode.

The target object is the first argument of the task's BDDL goal "(In <object> <region>)".
The object's free joint is moved by the shift in x and y; z, orientation stay; velocity is
zeroed; then sim.forward(). A direction is rejected if the new spot is closer than 0.06 m to
another object (0.13 m to the basket) or outside the box spanned by all objects at the start
of the episode plus 0.05 m; up to 20 seeded directions are tried.

The simulator is reached through the vector env's sub-envs (env.envs in LeRobot 0.6.1,
env._envs in LeRobot main), which exist only in the main process. So this script adds
--eval.use_async_envs=false unless that flag is given.

One JSON line per episode is appended to <output_dir>/shift_episodes.jsonl (env SHIFT_LOG
overrides the path). See README.md for the fields.
"""

import json
import math
import os
import sys
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from tqdm import trange

try:
    import lerobot_policy_griffin_alpha  # noqa: F401  (registers the griffin policy types)
    from lerobot_policy_griffin_alpha import GriffinAlphaPolicy
except ImportError:  # e.g. the Fast-WAM venv, which has no Griffin plugin
    GriffinAlphaPolicy = None

import lerobot.scripts.lerobot_eval as E  # noqa: E402
from lerobot.envs.utils import check_env_attributes_and_types, preprocess_observation
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import inside_slurm

# Newer LeRobot (the Fast-WAM venv) builds observations with prepare_canonical_observation and
# resets the processors each episode. LeRobot 0.6.1 (the Alpha-S venv) has neither, so these
# imports fail there and the Alpha-S path stays exactly as Griffin wrote it.
try:
    from lerobot.envs.utils import prepare_canonical_observation
except ImportError:
    prepare_canonical_observation = None
try:
    from lerobot.processor import bind_relative_anchor
except ImportError:
    bind_relative_anchor = None

_GRIFFIN_FLAGS = {
    "n_action_steps": "GRIFFIN_N_ACTION_STEPS",
    "fm_num_steps": "GRIFFIN_FM_NUM_STEPS",
    "delay": "GRIFFIN_DELAY",
}


def _pop_griffin_flags() -> dict[str, int]:
    """Remove ``--griffin.<name>=<int>`` from argv (draccus rejects unknown flags); return them."""
    values: dict[str, int] = {}
    kept = []
    for arg in sys.argv:
        if arg.startswith("--griffin."):
            name, _, raw = arg[len("--griffin.") :].partition("=")
            if name not in _GRIFFIN_FLAGS or not raw:
                known = ", ".join(_GRIFFIN_FLAGS)
                raise SystemExit(
                    f"unknown or valueless flag {arg!r}; known: --griffin.{{{known}}}=<int>"
                )
            values[name] = int(raw)
        else:
            kept.append(arg)
    sys.argv[:] = kept
    for name, env in _GRIFFIN_FLAGS.items():
        if name not in values and os.environ.get(env):
            values[name] = int(os.environ[env])
    return values


GRIFFIN = _pop_griffin_flags()

# ---------------------------------------------------------------------------------------------
# Shift settings and pure logic (no simulator needed; covered by test_shift_logic.py)
# ---------------------------------------------------------------------------------------------

SHIFT_MODES = ("none", "step0", "midreach", "blind")
MAX_TRIES = 20  # seeded directions tried before giving up on an episode
MIN_GAP = 0.06  # metres from the new spot to any other object's centre (xy)
BASKET_GAP = 0.13  # basket is ~0.17 m wide (mesh half-width 0.085) + ~0.045 for the object
BOX_MARGIN = 0.05  # allowed area = box around all objects at episode start, grown by this
TURN_RUN = 5  # consecutive steps that must head to the new spot for "turn time"
_SHIFT_TYPES = {"mode": str, "size": float, "radius": float, "seed": int}
_SHIFT_DEFAULTS = {"mode": "none", "size": 0.05, "radius": 0.15, "seed": 0}


def pop_shift_flags(argv: list[str]) -> tuple[dict, list[str]]:
    """Split ``--shift.<name>=<value>`` out of argv; return (settings, remaining argv)."""
    cfg = dict(_SHIFT_DEFAULTS)
    kept = []
    for arg in argv:
        if not arg.startswith("--shift."):
            kept.append(arg)
            continue
        name, _, raw = arg[len("--shift.") :].partition("=")
        if name not in _SHIFT_TYPES or not raw:
            known = ", ".join(_SHIFT_TYPES)
            raise SystemExit(f"unknown or valueless flag {arg!r}; known: --shift.{{{known}}}=...")
        cfg[name] = _SHIFT_TYPES[name](raw)
    if cfg["mode"] not in SHIFT_MODES:
        raise SystemExit(f"--shift.mode must be one of {SHIFT_MODES}, got {cfg['mode']!r}")
    if cfg["size"] <= 0 or cfg["radius"] <= 0 or cfg["seed"] < 0:
        raise SystemExit(f"--shift.size and --shift.radius must be > 0, seed >= 0: {cfg}")
    return cfg, kept


def force_sync_envs(argv: list[str]) -> list[str]:
    """Add --eval.use_async_envs=false unless the caller chose; sub-envs must be in-process."""
    if any(a.startswith("--eval.use_async_envs") for a in argv):
        return list(argv)
    return list(argv) + ["--eval.use_async_envs=false"]


def output_dir_from(argv: list[str]) -> str | None:
    for k, arg in enumerate(argv):
        if arg.startswith("--output_dir="):
            return arg.split("=", 1)[1]
        if arg == "--output_dir" and k + 1 < len(argv):
            return argv[k + 1]
    return None


def shift_directions(seed: int, task_id: int, init_index: int, n: int = MAX_TRIES) -> np.ndarray:
    """Unit xy directions (n, 2), fixed by (seed, task, start state) and nothing else."""
    rng = np.random.default_rng([int(seed), int(task_id), int(init_index)])
    angles = rng.uniform(0.0, 2.0 * math.pi, size=n)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def scene_box(xys, margin: float = BOX_MARGIN) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) around the given xy points, grown by margin."""
    a = np.asarray(list(xys), dtype=float).reshape(-1, 2)
    return (
        float(a[:, 0].min() - margin),
        float(a[:, 0].max() + margin),
        float(a[:, 1].min() - margin),
        float(a[:, 1].max() + margin),
    )


def pick_shift(old_xy, others, size, directions, box, gap=MIN_GAP, basket_gap=BASKET_GAP):
    """First direction whose new spot is in the box and clear of other objects.

    others: list of (name, xy). Returns (new_xy, direction_index) or (None, None).
    """
    old = np.asarray(old_xy, dtype=float)[:2]
    for k, u in enumerate(np.asarray(directions, dtype=float)):
        new = old + size * u
        if not (box[0] <= new[0] <= box[1] and box[2] <= new[1] <= box[3]):
            continue
        clear = True
        for name, xy in others:
            need = basket_gap if "basket" in name else gap
            if np.linalg.norm(new - np.asarray(xy, dtype=float)[:2]) < need:
                clear = False
                break
        if clear:
            return new, k
    return None, None


def trigger_now(mode: str, step: int, dist: float, radius: float) -> bool:
    if mode == "step0":
        return step == 0
    if mode in ("midreach", "blind"):
        return dist <= radius
    return False


def turn_time(trace, old_xyz, new_xyz, run: int = TURN_RUN) -> int | None:
    """Steps after the shift until the gripper starts `run` steps in a row that are each
    closer to the new spot than the old one AND closer to the new spot than the step before.

    trace[0] is the gripper at the shift (before any post-shift action), trace[k] after k steps.
    Returns the first such k (>= 1), or None.
    """
    t = np.asarray(trace, dtype=float).reshape(-1, 3)
    if len(t) < run + 1:
        return None
    d_new = np.linalg.norm(t - np.asarray(new_xyz, dtype=float), axis=1)
    d_old = np.linalg.norm(t - np.asarray(old_xyz, dtype=float), axis=1)
    good = np.zeros(len(t), dtype=bool)
    good[1:] = (d_new[1:] < d_old[1:]) & (d_new[1:] < d_new[:-1])
    for k in range(1, len(t) - run + 1):
        if good[k : k + run].all():
            return k
    return None


def image_paths(obs, prefix=()) -> list[tuple]:
    """Paths to batched image leaves (uint8 arrays with 4 dims) in a nested observation dict."""
    out = []
    for key, value in obs.items():
        if isinstance(value, dict):
            out += image_paths(value, prefix + (key,))
        elif isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 4:
            out.append(prefix + (key,))
    return out


def get_leaf(obs, path):
    for key in path:
        obs = obs[key]
    return obs


def paste_images(obs, i: int, images: dict) -> dict:
    """Write env i's row of each image leaf (path -> HWC array); copies the leaf first."""
    for path, image in images.items():
        leaf = np.array(get_leaf(obs, path))
        if leaf[i].shape != np.shape(image):
            raise RuntimeError(f"image {path}: row shape {leaf[i].shape} != {np.shape(image)}")
        leaf[i] = image
        get_leaf(obs, path[:-1])[path[-1]] = leaf
    return obs


# ---------------------------------------------------------------------------------------------
# Simulator access (LIBERO via LeRobot's LiberoEnv; cannot be tested without MuJoCo)
# ---------------------------------------------------------------------------------------------

_OBS_TIMING = ("_time_since_last_sample", "_sampled", "_current_delay")


class LiberoSim:
    """One LeRobot LiberoEnv sub-env: e._env is LIBERO's OffScreenRenderEnv, e._env.env the
    robosuite task env that owns sim, objects_dict, obj_body_id, parsed_problem and robots."""

    def __init__(self, lerobot_env, convert):
        self.e = lerobot_env
        self.robo = lerobot_env._env.env
        self.convert = convert  # raw LiberoEnv obs -> the format the vector env returns

    @property
    def task_id(self) -> int:
        return int(self.e.task_id)

    def init_index(self) -> int | None:
        """Start state used by the last reset (reset adds _reset_stride after using it)."""
        states = getattr(self.e, "_init_states", None)
        if states is None:
            return None
        return int((self.e.init_state_id - self.e._reset_stride) % len(states))

    def target(self) -> str:
        """First argument of the goal "(In <object> <region>)"; else first obj_of_interest."""
        names = self.robo.objects_dict
        for pred in self.robo.parsed_problem.get("goal_state", []):
            if len(pred) >= 2 and str(pred[0]).lower() == "in" and pred[1] in names:
                return pred[1]
        for name in getattr(self.robo, "obj_of_interest", []):
            if name in names:
                return name
        raise RuntimeError(f"no target object found in {self.robo.parsed_problem}")

    def object_xyz(self, name: str) -> np.ndarray:
        return np.array(self.robo.sim.data.body_xpos[self.robo.obj_body_id[name]], dtype=float)

    def objects(self) -> dict:
        return {name: self.object_xyz(name) for name in self.robo.objects_dict}

    def gripper_xyz(self) -> np.ndarray:
        site = self.robo.robots[0].eef_site_id
        if isinstance(site, dict):  # robosuite >= 1.5 keys it by arm
            site = next(iter(site.values()))
        return np.array(self.robo.sim.data.site_xpos[site], dtype=float)

    def move_object(self, name: str, new_xy) -> None:
        """Move the free joint so the body lands at new_xy; keep z and orientation; stop it."""
        sim = self.robo.sim
        joint = self.robo.objects_dict[name].joints[-1]
        qpos = np.array(sim.data.get_joint_qpos(joint), dtype=float)
        qpos[:2] += np.asarray(new_xy, dtype=float) - self.object_xyz(name)[:2]
        sim.data.set_joint_qpos(joint, qpos)
        sim.data.set_joint_qvel(joint, np.zeros_like(sim.data.get_joint_qvel(joint)))
        sim.forward()

    def render(self) -> dict:
        """Fresh observation of the current sim state, in the vector env's format.

        Forces robosuite's observables to re-sample, then restores their sampling clocks so
        the next env.step samples exactly as it would have.
        """
        observables = list(getattr(self.robo, "_observables", {}).values())
        saved = [(o, {k: getattr(o, k) for k in _OBS_TIMING if hasattr(o, k)}) for o in observables]
        raw = self.robo._get_observations(force_update=True)
        for o, fields in saved:
            for k, v in fields.items():
                setattr(o, k, v)
        return self.convert(self.e._format_raw_obs(raw))


def libero_sims(env):
    """LiberoSim per sub-env, or None if the sub-envs live in worker processes."""
    subs = getattr(env, "envs", None) or getattr(env, "_envs", None)
    if not subs or len(subs) != env.num_envs:
        return None
    convert = getattr(env, "_canonical_observation", None) or (lambda o: o)
    sims = []
    for e in subs:
        e = getattr(e, "unwrapped", e)
        inner = getattr(e, "_env", None)
        if inner is None or not hasattr(inner, "env") or not hasattr(e, "_format_raw_obs"):
            return None
        sims.append(LiberoSim(e, convert))
    return sims


# ---------------------------------------------------------------------------------------------
# Per-episode tracker: decides when to shift, does it, freezes images, and logs
# ---------------------------------------------------------------------------------------------


class ShiftTracker:
    """Called once after reset (step 0) and once after every env.step (step = steps taken)."""

    def __init__(self, sims, num_envs: int, cfg: dict):
        if sims is None and cfg["mode"] != "none":
            raise SystemExit(
                "--shift.mode needs in-process sub-envs to reach MuJoCo; run with "
                "--eval.use_async_envs=false (this LeRobot's vector env hides them)"
            )
        self.sims, self.cfg, self.n = sims, cfg, num_envs
        status = "none" if cfg["mode"] == "none" else "no_trigger"
        self.ep = [
            {"task_id": None, "init_state": None, "target": None, "status": status,
             "shift_step": None, "reach_step": None, "old_xy": None, "new_xy": None,
             "dir_index": None, "success": False, "steps": None}
            for _ in range(num_envs)
        ]
        self.box = [None] * num_envs
        self.trace = [None] * num_envs  # gripper xyz from the shift on
        self.frozen = [None] * num_envs  # blind: path -> pre-shift image
        self.old_new = [None] * num_envs
        self.finished = [False] * num_envs
        self.started = False

    def _start(self):
        for i, sim in enumerate(self.sims or []):
            ep = self.ep[i]
            ep["task_id"], ep["init_state"] = sim.task_id, sim.init_index()
            ep["target"] = sim.target()
            self.box[i] = scene_box(xyz[:2] for xyz in sim.objects().values())
            if ep["init_state"] is None and self.cfg["mode"] != "none":
                raise SystemExit("start-state index unknown (env.init_states off?); cannot seed")

    def observe(self, obs, step: int, done, successes=None):
        if not self.started:
            self.started = True
            try:
                self._start()
            except Exception as exc:  # none mode only logs; never let a sim read stop it
                if self.cfg["mode"] != "none":
                    raise
                print(f"[eval_shift] no sim access, logging without it: {exc!r}")
                self.sims = None
        for i in range(self.n):
            ep = self.ep[i]
            if self.finished[i]:
                continue
            if successes is not None and successes[i]:
                ep["success"] = True
            if done[i]:
                self.finished[i] = True
                ep["steps"] = step
                continue
            if self.sims is None:
                continue
            sim = self.sims[i]
            grip = sim.gripper_xyz()
            if self.trace[i] is not None:  # already shifted
                self.trace[i].append(grip)
                if self.frozen[i] is not None:
                    obs = paste_images(obs, i, self.frozen[i])
                continue
            target = sim.object_xyz(ep["target"])
            dist = float(np.linalg.norm(grip - target))
            if ep["reach_step"] is None and dist <= self.cfg["radius"]:
                ep["reach_step"] = step
            if ep["shift_step"] is not None or ep["status"] == "no_valid_shift":
                continue
            if trigger_now(self.cfg["mode"], step, dist, self.cfg["radius"]):
                obs = self._shift(obs, i, step, grip, target)
        return obs

    def _shift(self, obs, i, step, grip, target):
        sim, ep, cfg = self.sims[i], self.ep[i], self.cfg
        others = [(n, xyz[:2]) for n, xyz in sim.objects().items() if n != ep["target"]]
        dirs = shift_directions(cfg["seed"], ep["task_id"], ep["init_state"])
        new_xy, k = pick_shift(target[:2], others, cfg["size"], dirs, self.box[i])
        if new_xy is None:
            ep["status"] = "no_valid_shift"
            return obs
        paths = image_paths(obs)
        if not paths:
            raise RuntimeError("no uint8 image leaves found in the observation")
        if cfg["mode"] == "blind":  # images the policy saw just before the shift
            self.frozen[i] = {p: np.array(get_leaf(obs, p)[i]) for p in paths}
        sim.move_object(ep["target"], new_xy)
        if self.frozen[i] is None:  # the policy sees the moved object at once
            fresh = sim.render()
            obs = paste_images(obs, i, {p: get_leaf(fresh, p) for p in paths})
        ep.update(status="shifted", shift_step=step, dir_index=k)
        ep["old_xy"] = [round(float(v), 4) for v in target[:2]]
        ep["new_xy"] = [round(float(v), 4) for v in new_xy]
        self.old_new[i] = (target, np.array([new_xy[0], new_xy[1], target[2]]))
        self.trace[i] = [grip]
        return obs

    def records(self, extra: dict) -> list[dict]:
        out = []
        for i, ep in enumerate(self.ep):
            rec = dict(extra)
            rec.update(ep)
            rec["turn_time"] = None
            if self.trace[i] is not None:
                rec["turn_time"] = turn_time(self.trace[i], *self.old_new[i])
            out.append(rec)
        return out

    def write(self, path: str, extra: dict) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "a") as f:
            for rec in self.records(extra):
                f.write(json.dumps(rec) + "\n")


SHIFT, _argv = pop_shift_flags(sys.argv)
sys.argv[:] = force_sync_envs(_argv)
SHIFT_LOG = os.environ.get("SHIFT_LOG") or os.path.join(
    output_dir_from(sys.argv) or ".", "shift_episodes.jsonl"
)


def _chunk_kwargs(policy) -> dict:
    """``num_steps`` for the flow head when overridden; nothing otherwise (FAST takes none)."""
    if (
        GriffinAlphaPolicy is not None
        and isinstance(policy, GriffinAlphaPolicy)
        and GRIFFIN.get("fm_num_steps")
    ):
        return {"num_steps": GRIFFIN["fm_num_steps"]}
    return {}


def chunk_rollout(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seeds=None,
    return_observations=False,
    render_callback=None,
    **_ignored,
):
    """Drop-in for ``lerobot_eval.rollout``: predict a chunk, execute ``n_action_steps``, replan.

    Returns the same dict shape as the stock rollout. Extra keyword arguments newer lerobot
    versions pass (recording options) are accepted and ignored.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    policy.reset()
    new_lerobot = prepare_canonical_observation is not None
    if new_lerobot:  # mirror newer LeRobot's stock rollout
        preprocessor.reset()
        postprocessor.reset()
        if bind_relative_anchor is not None:
            bind_relative_anchor(policy, preprocessor)
    observation, info = env.reset(seed=seeds)
    prepare = (
        prepare_canonical_observation
        if new_lerobot and env.metadata.get("canonical_observations", False)
        else preprocess_observation
    )
    if render_callback is not None:
        render_callback(env)

    all_observations, all_actions, all_rewards, all_successes, all_dones = [], [], [], [], []
    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    n_action_steps = GRIFFIN.get("n_action_steps") or int(
        getattr(policy.config, "n_action_steps", 1)
    )
    delay = GRIFFIN.get("delay", 0)
    if delay < 0:
        raise SystemExit(f"--griffin.delay must be >= 0, got {delay}")
    progbar = trange(
        max_steps,
        desc=f"Chunk rollout ({n_action_steps}/replan, delay {delay}), at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    check_env_attributes_and_types(env)
    action_queue: deque = deque()  # env-ready actions, numpy (B, action_dim)
    tracker = ShiftTracker(libero_sims(env), env.num_envs, SHIFT)  # Test 2
    observation = tracker.observe(observation, 0, done)  # Test 2: step0 shift happens here

    while not np.all(done) and step < max_steps:
        proc_obs = prepare(observation)
        if return_observations:
            all_observations.append(deepcopy(proc_obs))

        # Replan on a fixed clock. With delay 0 the queue is empty exactly here, as in Griffin's.
        if step % n_action_steps == 0:
            first = step == 0
            if not first and len(action_queue) != delay:
                raise RuntimeError(
                    f"step {step}: {len(action_queue)} old actions queued, expected {delay}"
                )
            try:
                proc_obs["task"] = list(env.call("task_description"))
            except (AttributeError, NotImplementedError):
                try:
                    proc_obs["task"] = list(env.call("task"))
                except (AttributeError, NotImplementedError):
                    proc_obs["task"] = [""] * env.num_envs

            # Preprocess ONCE per chunk (this also caches the reference state for relative
            # actions).
            obs_for_policy = preprocessor(env_preprocessor(proc_obs))
            with torch.inference_mode():
                # (B, horizon, dim)
                chunk = policy.predict_action_chunk(obs_for_policy, **_chunk_kwargs(policy))
            end = delay + n_action_steps
            if chunk.shape[1] < end:
                raise SystemExit(
                    f"delay {delay} + n_action_steps {n_action_steps} = {end} "
                    f"exceeds the chunk length {chunk.shape[1]}"
                )
            chunk = chunk[:, :end]
            # Postprocess the WHOLE chunk against the single cached chunk-start state.
            # (B, chunk, env_dim)
            env_chunk = env_postprocessor({ACTION: postprocessor(chunk)})[ACTION]
            # The first `delay` actions of a new chunk belong to time already spent computing it,
            # so they are skipped; the old chunk's leftover actions (already queued) fill it.
            start = 0 if first else delay
            for k in range(start, end):
                action_queue.append(env_chunk[:, k, :].to("cpu").numpy())

        action_numpy = action_queue.popleft()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError("Unsupported `final_info` format; upgrade gymnasium to >= 1.0.")
            successes = final_info["is_success"].tolist()
        elif "is_success" in info:
            is_success = info["is_success"]
            successes = (
                is_success.tolist()
                if hasattr(is_success, "tolist")
                else [bool(is_success)] * env.num_envs
            )
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        observation = tracker.observe(observation, step + 1, done, successes)  # Test 2

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            torch.stack(all_successes, dim=1).any(dim=1).float().mean().item()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate * 100:.1f}%"})
        progbar.update()

    if return_observations:
        all_observations.append(deepcopy(prepare(observation)))

    tracker.write(  # Test 2: one JSON line per episode of this batch
        SHIFT_LOG,
        {"condition": SHIFT["mode"], "size": SHIFT["size"], "radius": SHIFT["radius"],
         "seed": SHIFT["seed"], "delay": delay, "n_action_steps": n_action_steps},
    )

    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        ret[OBS_STR] = {
            key: torch.stack([obs[key] for obs in all_observations], dim=1)
            for key in all_observations[0]
        }

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()
    return ret


E.rollout = chunk_rollout


if __name__ == "__main__":
    E.main()
