"""Check eval_shift's logic without LeRobot, MuJoCo or a GPU.

Stubs the LeRobot names the script imports (as test1/test_delay_schedule.py does), then:
  1. none mode: the action schedule matches eval_delay exactly (and test1's formula);
  2. flags: --shift.* parsed and stripped, async envs forced off;
  3. directions: deterministic per (seed, task, start state), identical across modes;
  4. rejection/redraw: box and gap rules, give-up case;
  5. turn time on synthetic gripper traces;
  6. full rollouts with fake sims: trigger fires once at the right step (per env in a batch),
     step0 fires at step 0, blind freezes images but not state, midreach shows the moved
     object at once, done envs are not shifted, JSONL lines are written.
Run: python3 -I test_shift_logic.py
"""

import json
import math
import os
import sys
import tempfile
import types

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
TEST1 = os.path.join(os.path.dirname(HERE), "test1")
sys.path.insert(0, HERE)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


for pkg in ["lerobot", "lerobot.scripts", "lerobot.envs", "lerobot.utils"]:
    _stub(pkg)
_stub("lerobot.scripts.lerobot_eval", rollout=None, main=lambda: None)
_stub(
    "lerobot.envs.utils",
    check_env_attributes_and_types=lambda env: None,
    preprocess_observation=lambda obs: {k: torch.as_tensor(v) for k, v in obs.items()},
)
_stub("lerobot.utils.constants", ACTION="action", OBS_STR="observation")
_stub("lerobot.utils.utils", inside_slurm=lambda: True)

N_STEPS, STRIDE, HORIZON = 100, 10, 50
LOG = os.path.join(tempfile.mkdtemp(), "shift_episodes.jsonl")


def load(module, argv, delay=0, folder=HERE):
    """(Re)import eval_shift / eval_delay with the given flags."""
    os.environ["GRIFFIN_DELAY"] = str(delay)
    os.environ["SHIFT_LOG"] = LOG
    sys.argv = ["x"] + list(argv)
    sys.modules.pop(module, None)
    sys.path.insert(0, folder)
    try:
        return __import__(module)
    finally:
        sys.path.remove(folder)


# ---------------------------------------------------------------- fakes


class FakeSim:
    """Gripper starts `start` m from the target along +x and closes `speed` m per step."""

    def __init__(self, task_id=3, init=1, start=0.40, speed=0.02):
        self.task_id, self.init, self.start, self.speed = task_id, init, start, speed
        self.t = 0
        self.objs = {
            "milk_1": np.array([0.0, 0.0, 0.9]),
            "basket_1": np.array([0.0, 0.30, 0.9]),
            "butter_1": np.array([0.15, -0.15, 0.9]),
            "ketchup_1": np.array([-0.15, 0.10, 0.9]),
        }
        self.moves = []
        self.target0 = self.objs["milk_1"].copy()

    def init_index(self):
        return self.init

    def target(self):
        return "milk_1"

    def objects(self):
        return {k: v.copy() for k, v in self.objs.items()}

    def object_xyz(self, name):
        return self.objs[name].copy()

    def gripper_xyz(self):
        d = max(self.start - self.speed * self.t, 0.0)
        return self.target0 + np.array([d, 0.0, 0.0])

    def move_object(self, name, new_xy):
        self.moves.append((self.t, name, np.array(new_xy)))
        self.objs[name][:2] = new_xy

    def render(self):
        return {"t": np.array([float(self.t)], dtype=np.float32),
                "img": np.full((4, 4, 3), 255, dtype=np.uint8)}


class FakeEnv:
    """Observation: 't' (live state) and 'img' (uint8 image whose pixels equal t mod 250)."""

    def __init__(self, num_envs, sims=None, done_at=None):
        self.num_envs = num_envs
        self.t = 0
        self.executed = []
        self.fake_sims = sims
        self.done_at = done_at or {}

    def call(self, name):
        if name == "_max_episode_steps":
            return [N_STEPS] * self.num_envs
        if name == "task_description":
            return ["task"] * self.num_envs
        raise AttributeError(name)

    def _obs(self):
        img = np.full((self.num_envs, 4, 4, 3), self.t % 250, dtype=np.uint8)
        return {"t": np.full((self.num_envs, 1), self.t, dtype=np.float32), "img": img}

    def reset(self, seed=None):
        self.t = 0
        for s in self.fake_sims or []:
            s.t = 0
        return self._obs(), {}

    def step(self, action):
        self.executed.append(action[0].copy())
        self.t += 1
        for s in self.fake_sims or []:
            s.t = self.t
        term = np.array([self.done_at.get(i) == self.t for i in range(self.num_envs)])
        return self._obs(), np.zeros(self.num_envs), term, np.zeros(self.num_envs, bool), {}


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(n_action_steps=STRIDE)

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        t = batch["t"][:, 0]
        b = t.shape[0]
        idx = torch.arange(HORIZON, dtype=torch.float32).expand(b, HORIZON)
        return torch.stack([t[:, None].expand(b, HORIZON), idx], dim=-1)


def rollout(mod, env, return_observations=False):
    ident = lambda x: x  # noqa: E731
    mod.libero_sims = lambda e: e.fake_sims
    return mod.chunk_rollout(env, FakePolicy(), ident, ident, ident, ident,
                             return_observations=return_observations)


def expected(s, d):
    if s < d + STRIDE:
        return (0, s)
    born = STRIDE * ((s - d) // STRIDE)
    return (born, s - born)


def read_log():
    with open(LOG) as f:
        return [json.loads(line) for line in f]


def clear_log():
    if os.path.exists(LOG):
        os.remove(LOG)


# ---------------------------------------------------------------- tests


def test_none_mode_matches_eval_delay():
    have_t1 = os.path.exists(os.path.join(TEST1, "eval_delay.py"))
    for d in [0, 2, 5, 10, 20, 40]:
        for b in [1, 2]:
            m = load("eval_shift", ["--shift.mode=none"], delay=d)
            env = FakeEnv(b)  # no sims: as with async envs, none mode must still run
            rollout(m, env)
            got = [(int(a[0]), int(a[1])) for a in env.executed]
            assert got == [expected(s, d) for s in range(N_STEPS)], f"delay {d}"
            if have_t1:
                ref = load("eval_delay", [], delay=d, folder=TEST1)
                env2 = FakeEnv(b)
                ident = lambda x: x  # noqa: E731
                ref.chunk_rollout(env2, FakePolicy(), ident, ident, ident, ident)
                assert all(np.array_equal(x, y) for x, y in zip(env.executed, env2.executed))
                assert len(env.executed) == len(env2.executed)
    # with fake sims attached, none mode reads them but changes nothing
    m = load("eval_shift", ["--shift.mode=none"], delay=5)
    env = FakeEnv(2, sims=[FakeSim(), FakeSim(init=2)])
    rollout(m, env)
    assert [(int(a[0]), int(a[1])) for a in env.executed] == [
        expected(s, 5) for s in range(N_STEPS)]
    assert all(not s.moves for s in env.fake_sims)
    print("none mode = eval_delay schedule (delays 0-40, batch 1 and 2)"
          + (", same actions as test1/eval_delay.py" if have_t1 else "") + ": OK")


def test_flags():
    m = load("eval_shift", ["--shift.mode=blind", "--shift.size=0.1", "--shift.seed=7",
                            "--griffin.delay=3", "--output_dir=out/x"])
    assert m.SHIFT == {"mode": "blind", "size": 0.1, "radius": 0.15, "seed": 7}
    assert m.GRIFFIN["delay"] == 3
    assert sys.argv == ["x", "--output_dir=out/x", "--eval.use_async_envs=false"], sys.argv
    assert m.output_dir_from(["--output_dir", "a/b"]) == "a/b"
    assert m.force_sync_envs(["--eval.use_async_envs=true"]) == ["--eval.use_async_envs=true"]
    for bad in (["--shift.mode=sideways"], ["--shift.size=0"], ["--shift.colour=1"]):
        try:
            m.pop_shift_flags(bad)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")
    print("flags parsed, stripped, async forced off, bad values refused: OK")


def test_directions_deterministic():
    m = load("eval_shift", [])
    a = m.shift_directions(0, 3, 1)
    assert np.array_equal(a, m.shift_directions(0, 3, 1))
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert not np.allclose(a, m.shift_directions(0, 3, 2))
    assert not np.allclose(a, m.shift_directions(0, 4, 1))
    assert not np.allclose(a, m.shift_directions(1, 3, 1))
    # across modes, the same episode gets the same direction
    dirs = {}
    for mode in ["step0", "midreach", "blind"]:
        clear_log()
        mm = load("eval_shift", [f"--shift.mode={mode}", "--shift.size=0.05"])
        rollout(mm, FakeEnv(1, sims=[FakeSim()]))
        rec = read_log()[0]
        delta = np.subtract(rec["new_xy"], rec["old_xy"])
        dirs[mode] = (rec["dir_index"], round(math.degrees(math.atan2(delta[1], delta[0])), 3))
    assert len(set(dirs.values())) == 1, dirs
    print(f"directions deterministic per (seed, task, init), same in every mode {dirs}: OK")


def test_rejection():
    m = load("eval_shift", [])
    box = (-1.0, 1.0, -1.0, 1.0)
    dirs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    # an object 0.10 m along +x blocks dir 0 (new spot 0.0 m from it)
    new, k = m.pick_shift([0, 0], [("butter_1", [0.10, 0.0])], 0.10, dirs, box)
    assert k == 1 and np.allclose(new, [0, 0.10]), (new, k)
    # 0.07 m away is clear (gap 0.06), 0.05 m is not
    _, k = m.pick_shift([0, 0], [("butter_1", [0.10, 0.07])], 0.10, dirs, box)
    assert k == 0
    _, k = m.pick_shift([0, 0], [("butter_1", [0.10, 0.05])], 0.10, dirs, box)
    assert k == 1
    # the basket needs 0.13 m
    _, k = m.pick_shift([0, 0], [("basket_1", [0.10, -0.12])], 0.10, dirs, box)
    assert k == 1
    # box: +x and +y leave the box, so -x is chosen
    _, k = m.pick_shift([0, 0], [], 0.10, dirs, (-1.0, 0.05, -1.0, 0.05))
    assert k == 2
    # nothing fits
    new, k = m.pick_shift([0, 0], [], 0.10, dirs, (-0.05, 0.05, -0.05, 0.05))
    assert new is None and k is None
    assert m.scene_box([[0, 0], [1, 2]], 0.5) == (-0.5, 1.5, -0.5, 2.5)
    # in a rollout: an episode with no valid spot logs no_valid_shift and moves nothing
    clear_log()
    mm = load("eval_shift", ["--shift.mode=midreach", "--shift.size=0.10"])
    sim = FakeSim()
    sim.objs = {"milk_1": np.array([0.0, 0.0, 0.9]), "basket_1": np.array([0.02, 0.0, 0.9])}
    rollout(mm, FakeEnv(1, sims=[sim]))
    rec = read_log()[0]
    assert rec["status"] == "no_valid_shift" and rec["shift_step"] is None and not sim.moves
    print("rejection by gap / basket gap / box, redraw order, give-up logged: OK")


def test_turn_time():
    m = load("eval_shift", [])
    old, new = np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])
    # homes in on the new spot from x=-0.2 (x_k = 0.1 - 0.3 * 0.85^k, always approaching);
    # it is closer to new than old once x > 0.05, i.e. 0.85^k < 1/6, first at k = 12
    trace = [[0.1 - 0.3 * 0.85**k, 0.0, 0.1] for k in range(30)]
    assert m.turn_time(trace, old, new) == 12, m.turn_time(trace, old, new)
    # goes to the old spot: never
    trace = [[-0.2 + 0.02 * k, 0.0, 0.1] for k in range(10)] + [[0.0, 0.0, 0.1]] * 10
    assert m.turn_time(trace, old, new) is None
    # crosses, backs off at k = 3, then approaches for 5 steps from k = 4
    xs = [0.06, 0.07, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095]
    trace = [[0.0, 0.0, 0.1]] + [[x, 0.0, 0.1] for x in xs]
    assert m.turn_time(trace, old, new) == 4, m.turn_time(trace, old, new)
    # only 4 good steps in a row is not enough
    assert m.turn_time(trace[:8], old, new) is None
    # too short to judge
    assert m.turn_time([[0.09, 0, 0]] * 3, old, new) is None
    print("turn time on synthetic traces: OK")


def test_rollouts():
    # midreach: gripper starts 0.40 m away, closes 0.02 m/step -> within 0.15 m at step 13
    for mode in ["midreach", "blind"]:
        clear_log()
        m = load("eval_shift", [f"--shift.mode={mode}", "--shift.size=0.05"], delay=2)
        sims = [FakeSim(start=0.40, speed=0.02), FakeSim(init=2, start=0.40, speed=0.01)]
        want = [next(t for t in range(N_STEPS) if s.start - s.speed * t <= 0.15) for s in sims]
        assert want[0] == 13 and want[1] in (25, 26), want  # 26: 0.40 - 0.25 > 0.15 in floats
        env = FakeEnv(2, sims=sims)
        ret = rollout(m, env, return_observations=True)
        # schedule unaffected by the shift
        assert [(int(a[0]), int(a[1])) for a in env.executed] == [
            expected(s, 2) for s in range(N_STEPS)]
        recs = read_log()
        assert [r["shift_step"] for r in recs] == want, recs  # each env on its own
        assert all(len(s.moves) == 1 for s in sims)  # fires once
        assert [s.moves[0][0] for s in sims] == want
        img = ret["observation"]["img"].numpy()[..., 0, 0, 0]  # (B, steps + 1)
        tt = ret["observation"]["t"].numpy()[..., 0]
        for i, s in enumerate(want):
            assert (tt[i] == np.arange(N_STEPS + 1)).all()  # state stays live
            assert (img[i, :s] == np.arange(s) % 250).all()
            if mode == "blind":
                # frozen pre-shift frame for every step the policy can still act on (the
                # last entry is the terminal observation, which is never fed to the policy)
                assert (img[i, s:N_STEPS] == s).all(), img[i, s:N_STEPS]
            else:
                assert img[i, s] == 255  # re-rendered right after the shift
                assert (img[i, s + 1:] == np.arange(s + 1, N_STEPS + 1) % 250).all()
        r = recs[0]
        assert r["condition"] == mode and r["delay"] == 2 and r["task_id"] == 3
        assert r["init_state"] == 1 and recs[1]["init_state"] == 2
        assert abs(np.linalg.norm(np.subtract(r["new_xy"], r["old_xy"])) - 0.05) < 1e-3
        assert r["reach_step"] == 13 and r["steps"] == N_STEPS and r["success"] is False
    # step0: shift at step 0, before the first action; first chunk sees the moved object
    clear_log()
    m = load("eval_shift", ["--shift.mode=step0", "--shift.size=0.02"])
    sims = [FakeSim()]
    ret = rollout(m, FakeEnv(1, sims=sims), return_observations=True)
    assert read_log()[0]["shift_step"] == 0 and sims[0].moves[0][0] == 0
    assert ret["observation"]["img"].numpy()[0, 0, 0, 0, 0] == 255
    # an env that finishes before reaching is never shifted; steps recorded
    clear_log()
    m = load("eval_shift", ["--shift.mode=midreach"])
    sims = [FakeSim(), FakeSim(init=2)]
    rollout(m, FakeEnv(2, sims=sims, done_at={0: 7}))
    recs = read_log()
    assert recs[0]["status"] == "no_trigger" and recs[0]["steps"] == 7 and not sims[0].moves
    assert recs[1]["status"] == "shifted" and recs[1]["steps"] == N_STEPS
    # shift modes refuse to run without in-process sims
    m = load("eval_shift", ["--shift.mode=midreach"])
    try:
        rollout(m, FakeEnv(1))
    except SystemExit:
        pass
    else:
        raise AssertionError("ran without sims")
    print("rollouts: trigger once per env at the right step, step0, blind freezes images "
          "not state, done envs skipped, JSONL written: OK")


if __name__ == "__main__":
    test_none_mode_matches_eval_delay()
    test_flags()
    test_directions_deterministic()
    test_rejection()
    test_turn_time()
    test_rollouts()
    print("ALL PASSED")
