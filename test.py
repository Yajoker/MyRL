"""Test script for MYRL hierarchical navigation best models.

Location: this file is intended to live in the `myrl/` repo folder
(i.e. `~/Work/YF/Navigation/myrl/`).

It loads the latest (best) high-level and low-level checkpoints from:
- myrl/myrl/models/best_models_7.24/best_models/high_level
- myrl/myrl/models/best_models_7.24/best_models/low_level

and evaluates them in the IR-SIM wrapper used by this repo.

Usage examples:
    python test.py
    python test.py --episodes 600 --max-steps 400 --disable-plotting
    python test.py --world-file worlds/eval_world.yaml
    python test.py --seeds 0,1,2,3,4,5,6,7,8,9 --no-verbose-step-log
    MYRL_PER_SEED_LOGS=1 python test.py   # 10 seeds + myrl/logs_2/test_<world>_<ts>_seed<N>.log
    python test.py --per-seed-logs   # 默认写入 myrl/logs_2
    python test.py --parallel-seeds  # 10 进程并行，仅子日志（logs_2），无总日志

Notes:
- This script is robust to current working directory.
- In headless environments, use `--disable-plotting` (default auto).
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# Ensure local imports work regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # .../Navigation
IRSIM_SRC_ROOT = REPO_ROOT / "ir-sim"  # contains the `irsim/` package

for p in (SCRIPT_DIR, REPO_ROOT, IRSIM_SRC_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# headless 环境下避免 matplotlib 反复尝试 Tk/Qt 后端（即使本脚本默认不画图）
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

from integration import HierarchicalNavigationSystem
from robot_nav.SIM_ENV.sim import SIM


_EPOCH_RE = re.compile(r"epoch(?P<epoch>\d+)")


# =========================
# Model loading configuration
# =========================
# Edit this section when you need to change where checkpoints are loaded from.
# Recommended: only change EXPLICIT_MODEL_EPOCH (e.g. "009") to lock both high/low models.
# Priority: explicit prefix > explicit epoch > auto latest.
MODEL_ROOT = (SCRIPT_DIR / "myrl" / "models" / "best_models_7.24").resolve()
EXPLICIT_BEST_MODELS_ROOT = MODEL_ROOT / "best_models"
# >>> ONLY CHANGE THIS VALUE <<<
# Example: MODEL_EPOCH_INDEX = "009"
# Use None to auto-select latest checkpoints.
MODEL_EPOCH_INDEX: Optional[str] = "058"
EXPLICIT_MODEL_EPOCH: Optional[str] = MODEL_EPOCH_INDEX
EXPLICIT_HIGH_LEVEL_PREFIX: Optional[str] = None
EXPLICIT_LOW_LEVEL_PREFIX: Optional[str] = None

# 多随机种子评估：未指定 --seed / --seeds 且启用 --per-seed-logs（或 MYRL_PER_SEED_LOGS=1）时使用。
DEFAULT_TEN_EVAL_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# 分种子测试日志默认目录（与 --per-seed-log-dir 默认值一致；即 myrl/logs_2）
DEFAULT_PER_SEED_LOG_DIR: Path = (SCRIPT_DIR / "logs_2").resolve()

# --log-to-file 时保持文件句柄存活，避免被 GC 关闭
_LOG_FILE_HANDLE: Optional[object] = None


def _redirect_stdio_to_file(path: Path) -> None:
    global _LOG_FILE_HANDLE
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE_HANDLE = open(path, "w", encoding="utf-8")
    sys.stdout = _LOG_FILE_HANDLE  # type: ignore[assignment]
    sys.stderr = _LOG_FILE_HANDLE  # type: ignore[assignment]


def _filtered_argv_for_parallel_child() -> list[str]:
    """从 sys.argv 去掉仅父进程使用的参数，供子进程复用。"""
    argv = sys.argv[1:]
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--parallel-seeds":
            i += 1
            continue
        if a == "--per-seed-logs":
            i += 1
            continue
        if a == "--seed" and i + 1 < len(argv):
            i += 2
            continue
        if a.startswith("--seed="):
            i += 1
            continue
        if a == "--seeds" and i + 1 < len(argv):
            i += 2
            continue
        if a.startswith("--seeds="):
            i += 1
            continue
        if a == "--log-to-file" and i + 1 < len(argv):
            i += 2
            continue
        if a.startswith("--log-to-file="):
            i += 1
            continue
        if a == "--per-seed-log-dir" and i + 1 < len(argv):
            out.extend([a, argv[i + 1]])
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def _run_parallel_seed_workers(args: argparse.Namespace) -> None:
    """启动多个子进程，各写独立日志；父进程不加载模型、几乎不向 stdout 输出。"""
    if args.seed is not None:
        raise ValueError("--parallel-seeds 不能与 --seed 同时使用（每个子进程会自带 --seed）")

    world_file = _resolve_world_file(args.world_file)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.per_seed_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds_csv(args.seeds) if args.seeds is not None else list(DEFAULT_TEN_EVAL_SEEDS)

    env = os.environ.copy()
    env.pop("MYRL_PER_SEED_LOGS", None)

    script_path = Path(__file__).resolve()
    base_args = _filtered_argv_for_parallel_child()
    procs: List[subprocess.Popen] = []
    for seed in seeds:
        log_path = log_dir / f"test_{world_file.stem}_{run_ts}_seed{seed}.log"
        cmd = [sys.executable, str(script_path)] + base_args + ["--seed", str(seed), "--log-to-file", str(log_path)]
        procs.append(
            subprocess.Popen(
                cmd,
                env=env,
                cwd=str(SCRIPT_DIR),
                stdout=subprocess.DEVNULL,
                stderr=None,
            )
        )
    exit_codes = [p.wait() for p in procs]
    if any(c != 0 for c in exit_codes):
        raise RuntimeError(f"部分子进程失败 exit codes={exit_codes}")


class _TeeStream:
    """Duplicate writes to multiple text streams (e.g. nohup log + per-seed file)."""

    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


@dataclass(frozen=True)
class BestModelSelection:
    high_prefix: str
    low_prefix: str
    high_dir: Path
    low_dir: Path


@dataclass(frozen=True)
class EvalRunSummary:
    """Aggregated metrics for one full test run (one seed)."""

    goals: int
    collisions: int
    timeouts: int
    test_scenarios: int
    total_steps: int
    goal_rate: float
    collision_rate: float
    timeout_rate: float
    avg_ep_len: float
    avg_step_rew: float
    avg_step_rew_std: float
    avg_ep_rew: float
    avg_ep_rew_std: float
    avg_stg: float
    std_stg: float
    mean_lin: float
    std_lin: float
    mean_ang: float
    std_ang: float


def _set_all_seeds(seed: int) -> None:
    """Align numpy / Python / torch RNGs for reproducible evaluation."""

    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_seeds_csv(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--seeds must list at least one integer, e.g. 0,1,2,3")
    out: list[int] = []
    for p in parts:
        out.append(int(p))
    return out


def _parse_epoch(name: str) -> Optional[int]:
    m = _EPOCH_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group("epoch"))
    except Exception:
        return None


def _resolve_best_models_root() -> Path:
    """Resolve best_models directory robustly."""

    candidates = [
        # explicit config (preferred)
        EXPLICIT_BEST_MODELS_ROOT,
        # common layout: myrl/test_best_models.py + myrl/myrl/models/best_models
        SCRIPT_DIR / "myrl" / "models" / "best_models",
        # fallback if running from inside package folder
        SCRIPT_DIR / "models" / "best_models",
        # cwd-based fallbacks
        Path.cwd() / "myrl" / "models" / "best_models",
        Path.cwd() / "models" / "best_models",
    ]

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    return candidates[0]


def _pick_latest_high_level_prefix(high_dir: Path) -> str:
    """Pick latest high-level prefix that has both _A and _B."""

    a_files = sorted(high_dir.glob("*_A.pth"))
    groups = []
    for a_path in a_files:
        base = a_path.name[: -len("_A.pth")]
        b_path = high_dir / f"{base}_B.pth"
        if not b_path.exists():
            continue

        epoch = _parse_epoch(base)
        # Score: (epoch if present else -1, mtime)
        mtime = max(a_path.stat().st_mtime, b_path.stat().st_mtime)
        groups.append((epoch if epoch is not None else -1, mtime, base))

    if not groups:
        raise FileNotFoundError(
            f"No paired high-level checkpoints found in: {high_dir}\n"
            "Expected files like: high_level_planner_..._A.pth and _B.pth"
        )

    groups.sort(key=lambda x: (x[0], x[1]))
    return groups[-1][2]


def _pick_latest_low_level_prefix(low_dir: Path) -> str:
    """Pick latest low-level prefix that has actor/actor_target/critic/critic_target."""

    actor_files = sorted(low_dir.glob("*_actor.pth"))
    groups = []
    for actor_path in actor_files:
        base = actor_path.name[: -len("_actor.pth")]
        needed = [
            low_dir / f"{base}_actor.pth",
            low_dir / f"{base}_actor_target.pth",
            low_dir / f"{base}_critic.pth",
            low_dir / f"{base}_critic_target.pth",
        ]
        if not all(p.exists() for p in needed):
            continue

        epoch = _parse_epoch(base)
        mtime = max(p.stat().st_mtime for p in needed)
        groups.append((epoch if epoch is not None else -1, mtime, base))

    if not groups:
        raise FileNotFoundError(
            f"No complete low-level checkpoint set found in: {low_dir}\n"
            "Expected files like: low_level_controller_..._actor.pth / _critic.pth / _actor_target.pth / _critic_target.pth"
        )

    groups.sort(key=lambda x: (x[0], x[1]))
    return groups[-1][2]


def _pick_high_level_prefix_by_epoch(high_dir: Path, epoch_value: str) -> str:
    """Pick high-level prefix for a given epoch (e.g. '009')."""

    target_epoch = int(epoch_value)
    a_files = sorted(high_dir.glob("*_A.pth"))
    groups = []
    for a_path in a_files:
        base = a_path.name[: -len("_A.pth")]
        b_path = high_dir / f"{base}_B.pth"
        if not b_path.exists():
            continue
        if _parse_epoch(base) != target_epoch:
            continue
        mtime = max(a_path.stat().st_mtime, b_path.stat().st_mtime)
        groups.append((mtime, base))

    if not groups:
        raise FileNotFoundError(
            f"No high-level checkpoint pair found for epoch {epoch_value} in: {high_dir}"
        )

    groups.sort(key=lambda x: x[0])
    return groups[-1][1]


def _pick_low_level_prefix_by_epoch(low_dir: Path, epoch_value: str) -> str:
    """Pick low-level prefix for a given epoch (e.g. '009')."""

    target_epoch = int(epoch_value)
    actor_files = sorted(low_dir.glob("*_actor.pth"))
    groups = []
    for actor_path in actor_files:
        base = actor_path.name[: -len("_actor.pth")]
        needed = [
            low_dir / f"{base}_actor.pth",
            low_dir / f"{base}_actor_target.pth",
            low_dir / f"{base}_critic.pth",
            low_dir / f"{base}_critic_target.pth",
        ]
        if not all(p.exists() for p in needed):
            continue
        if _parse_epoch(base) != target_epoch:
            continue
        mtime = max(p.stat().st_mtime for p in needed)
        groups.append((mtime, base))

    if not groups:
        raise FileNotFoundError(
            f"No complete low-level checkpoint set found for epoch {epoch_value} in: {low_dir}"
        )

    groups.sort(key=lambda x: x[0])
    return groups[-1][1]


def _load_low_level_weights_map_location(
    system: HierarchicalNavigationSystem,
    low_dir: Path,
    prefix: str,
    device: torch.device,
) -> None:
    """Load the complete low-level checkpoint using its strict contract."""

    del device
    system.low_level_controller.load_model(prefix, low_dir)


def _resolve_world_file(world_file_arg: Optional[str]) -> Path:
    if world_file_arg:
        p = Path(world_file_arg)
        if not p.is_absolute():
            # allow relative to script dir and CWD
            for base in [SCRIPT_DIR, Path.cwd()]:
                cand = (base / p).resolve()
                if cand.exists():
                    return cand
        return p

    # default
    candidates = [
        #SCRIPT_DIR / "worlds" / "eval_2.5_12.yaml",
        #Path.cwd() / "worlds" / "eval_2.5_12.yaml",
        #SCRIPT_DIR / "worlds" / "env3_6.yaml",
        #Path.cwd() / "worlds" / "env3_6.yaml",
        SCRIPT_DIR / "worlds" / "eval_2_2.5_16.yaml",
        Path.cwd() / "worlds" / "eval_2_2.5_16.yaml",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return candidates[0]


def _get_robot_pose(sim: SIM) -> Tuple[float, float, float]:
    robot_state = sim.env.get_robot_state()
    return (
        float(robot_state[0].item()),
        float(robot_state[1].item()),
        float(robot_state[2].item()),
    )


def _auto_disable_plotting(cli_value: Optional[bool]) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    # auto: disable plotting if headless
    return not bool(os.environ.get("DISPLAY"))


def _mean_std(x: np.ndarray) -> Tuple[float, float]:
    if x.size <= 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))


def _evaluate_scenarios(
    system: HierarchicalNavigationSystem,
    sim: SIM,
    test_scenarios: int,
    max_steps: int,
    verbose_step_log: bool,
    step_log_every: int,
    use_tqdm_progress: bool,
) -> EvalRunSummary:
    """Run all test scenarios once (caller must set RNG seeds before this)."""

    total_rewards: list[float] = []
    ep_rewards: list[float] = []
    lin_actions: list[float] = []
    ang_actions: list[float] = []

    total_steps = 0
    collisions = 0
    goals = 0
    timeouts = 0
    steps_to_goal: list[int] = []

    scenario_iter = range(test_scenarios)
    if use_tqdm_progress and (not verbose_step_log):
        if tqdm is None:
            print("[WARN] tqdm not installed; falling back to plain loop.")
        else:
            scenario_iter = tqdm.tqdm(scenario_iter)

    for scenario_idx in scenario_iter:
        system.reset()
        latest_scan, distance, cos, sin, collision, goal, _a, reward = sim.reset()
        packet = system.prepare_observation(
            sim.get_last_lidar_observation()
        )

        ep_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            lin_cmd, ang_cmd = system.step(
                packet,
                goal_distance=distance,
                goal_cos=cos,
                goal_sin=sin,
            )

            lin_actions.append(float(lin_cmd))
            ang_actions.append(float(ang_cmd))

            latest_scan, distance, cos, sin, collision, goal, _a, reward = sim.step(
                lin_velocity=float(lin_cmd),
                ang_velocity=float(ang_cmd),
            )
            packet_next = system.prepare_observation(
                sim.get_last_lidar_observation()
            )

            ep_reward += float(reward)
            total_rewards.append(float(reward))
            total_steps += 1
            steps += 1

            if verbose_step_log and (steps == 1 or steps % step_log_every == 0):
                print(
                    f"Test | Scenario {scenario_idx+1:4d}/{test_scenarios} | "
                    f"Step {steps:3d}/{max_steps} | Reward: {reward:7.2f} | Dist: {distance:6.2f}"
                )

            done = bool(collision) or bool(goal)
            packet = packet_next

            if done:
                if bool(collision):
                    collisions += 1
                if bool(goal):
                    goals += 1
                    steps_to_goal.append(steps)

        if not done and steps >= max_steps:
            timeouts += 1

        ep_rewards.append(ep_reward)

        status = "🎯 GOAL" if goal else "💥 COLLISION" if collision else "⏰ TIMEOUT"
        print(
            f"Scenario {scenario_idx+1:4d} finished: {status} | "
            f"Steps: {steps:3d} | Total Reward: {ep_reward:8.1f}"
        )

    total_rewards_arr = np.array(total_rewards, dtype=np.float32) if total_rewards else np.array([0.0], dtype=np.float32)
    ep_rewards_arr = np.array(ep_rewards, dtype=np.float32) if ep_rewards else np.array([0.0], dtype=np.float32)
    lin_actions_arr = np.array(lin_actions, dtype=np.float32) if lin_actions else np.array([0.0], dtype=np.float32)
    ang_actions_arr = np.array(ang_actions, dtype=np.float32) if ang_actions else np.array([0.0], dtype=np.float32)

    avg_ep_len = total_steps / test_scenarios
    goal_rate = goals / test_scenarios
    collision_rate = collisions / test_scenarios
    timeout_rate = timeouts / test_scenarios

    avg_step_rew, avg_step_rew_std = _mean_std(total_rewards_arr)
    avg_ep_rew, avg_ep_rew_std = _mean_std(ep_rewards_arr)
    mean_lin, std_lin = _mean_std(lin_actions_arr)
    mean_ang, std_ang = _mean_std(ang_actions_arr)

    if steps_to_goal:
        stg_arr = np.array(steps_to_goal, dtype=np.float32)
        avg_stg, std_stg = _mean_std(stg_arr)
    else:
        avg_stg, std_stg = float("nan"), float("nan")

    return EvalRunSummary(
        goals=goals,
        collisions=collisions,
        timeouts=timeouts,
        test_scenarios=test_scenarios,
        total_steps=total_steps,
        goal_rate=goal_rate,
        collision_rate=collision_rate,
        timeout_rate=timeout_rate,
        avg_ep_len=avg_ep_len,
        avg_step_rew=avg_step_rew,
        avg_step_rew_std=avg_step_rew_std,
        avg_ep_rew=avg_ep_rew,
        avg_ep_rew_std=avg_ep_rew_std,
        avg_stg=avg_stg,
        std_stg=std_stg,
        mean_lin=mean_lin,
        std_lin=std_lin,
        mean_ang=mean_ang,
        std_ang=std_ang,
    )


def _print_run_summary(summary: EvalRunSummary, seed_used: int) -> None:
    s = summary
    print("=" * 70)
    print("SUMMARY")
    print(f"  [Seed]          {seed_used}")
    print(f"  Goal rate:       {s.goal_rate:.3f} ({s.goals}/{s.test_scenarios})")
    print(f"  Collision rate:  {s.collision_rate:.3f} ({s.collisions}/{s.test_scenarios})")
    print(f"  Timeout rate:    {s.timeout_rate:.3f} ({s.timeouts}/{s.test_scenarios})")
    print(f"  Avg ep length:   {s.avg_ep_len:.1f} steps")
    print(f"  Avg step reward: {s.avg_step_rew:.4f} ± {s.avg_step_rew_std:.4f}")
    print(f"  Avg ep reward:   {s.avg_ep_rew:.4f} ± {s.avg_ep_rew_std:.4f}")
    print(f"  Steps-to-goal:   {s.avg_stg:.2f} ± {s.std_stg:.2f} (on successful eps)")
    print(f"  Mean lin cmd:    {s.mean_lin:.4f} ± {s.std_lin:.4f}")
    print(f"  Mean ang cmd:    {s.mean_ang:.4f} ± {s.std_ang:.4f}")
    print("=" * 70)


def _print_multi_seed_aggregate(summaries: list[EvalRunSummary], seeds: list[int]) -> None:
    if len(summaries) < 2:
        return
    arr = lambda key: np.array([getattr(s, key) for s in summaries], dtype=np.float64)
    # Skip nan-heavy fields for aggregate if any run had no goals
    stg_vals = [s.avg_stg for s in summaries if not np.isnan(s.avg_stg)]

    print("=" * 70)
    print(f"MULTI-SEED AGGREGATE  (n={len(summaries)} seeds: {seeds})")
    for label, key in [
        ("Goal rate", "goal_rate"),
        ("Collision rate", "collision_rate"),
        ("Timeout rate", "timeout_rate"),
        ("Avg ep length", "avg_ep_len"),
        ("Avg step reward", "avg_step_rew"),
        ("Avg ep reward", "avg_ep_rew"),
    ]:
        a = arr(key)
        print(f"  {label:16s} mean {float(a.mean()):.4f}  std {float(a.std(ddof=1)):.4f}")
    if len(stg_vals) >= 2:
        stg_a = np.array(stg_vals, dtype=np.float64)
        print(f"  {'Steps-to-goal':16s} mean {float(stg_a.mean()):.4f}  std {float(stg_a.std(ddof=1)):.4f} (over seeds with ≥1 goal)")
    elif len(stg_vals) == 1:
        print(f"  {'Steps-to-goal':16s} (only one seed had goals; skipped aggregate std)")
    print("=" * 70)


def _select_best_models() -> BestModelSelection:
    best_root = _resolve_best_models_root()
    high_dir = best_root / "high_level"
    low_dir = best_root / "low_level"

    if not high_dir.exists() or not low_dir.exists():
        raise FileNotFoundError(
            "best_models 目录结构不存在或不完整\n"
            f"Resolved best_root: {best_root}\n"
            f"Expected: {high_dir} and {low_dir}"
        )

    if EXPLICIT_HIGH_LEVEL_PREFIX is not None:
        high_prefix = EXPLICIT_HIGH_LEVEL_PREFIX
    elif EXPLICIT_MODEL_EPOCH is not None:
        high_prefix = _pick_high_level_prefix_by_epoch(high_dir, EXPLICIT_MODEL_EPOCH)
    else:
        high_prefix = _pick_latest_high_level_prefix(high_dir)

    if EXPLICIT_LOW_LEVEL_PREFIX is not None:
        low_prefix = EXPLICIT_LOW_LEVEL_PREFIX
    elif EXPLICIT_MODEL_EPOCH is not None:
        low_prefix = _pick_low_level_prefix_by_epoch(low_dir, EXPLICIT_MODEL_EPOCH)
    else:
        low_prefix = _pick_latest_low_level_prefix(low_dir)

    # Fast fail for manual prefix configuration mistakes.
    high_a = high_dir / f"{high_prefix}_A.pth"
    high_b = high_dir / f"{high_prefix}_B.pth"
    if not high_a.exists() or not high_b.exists():
        raise FileNotFoundError(
            "High-level prefix invalid or incomplete\n"
            f"prefix: {high_prefix}\n"
            f"expected: {high_a} and {high_b}"
        )

    low_needed = [
        low_dir / f"{low_prefix}_actor.pth",
        low_dir / f"{low_prefix}_actor_target.pth",
        low_dir / f"{low_prefix}_critic.pth",
        low_dir / f"{low_prefix}_critic_target.pth",
    ]
    if not all(p.exists() for p in low_needed):
        raise FileNotFoundError(
            "Low-level prefix invalid or incomplete\n"
            f"prefix: {low_prefix}\n"
            f"expected files: {', '.join(str(p) for p in low_needed)}"
        )

    return BestModelSelection(
        high_prefix=high_prefix,
        low_prefix=low_prefix,
        high_dir=high_dir,
        low_dir=low_dir,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MYRL best models")
    # Keep compatibility with older naming, but align with test_SAC terminology.
    p.add_argument(
        "--test-scenarios",
        type=int,
        default=600,
        help="Number of test scenarios (episodes) to evaluate",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Alias of --test-scenarios (kept for backward compatibility)",
    )
    p.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    p.add_argument("--world-file", type=str, default=None, help="Path to IR-SIM yaml world")
    p.add_argument(
        "--disable-plotting",
        action="store_true",
        default=None,
        help="Disable IR-SIM rendering/plotting (recommended for headless)",
    )
    p.add_argument(
        "--enable-plotting",
        action="store_true",
        default=None,
        help="Force enable rendering/plotting",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed (numpy / random / torch); ignored if --seeds is set")
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for repeated full test runs, e.g. 0,1,2,3,4,5,6,7,8,9; prints per-seed SUMMARY and a MULTI-SEED AGGREGATE block",
    )
    p.add_argument(
        "--per-seed-logs",
        action="store_true",
        help="Use DEFAULT_TEN_EVAL_SEEDS when no --seed/--seeds, and write one log file per seed under --per-seed-log-dir (also set MYRL_PER_SEED_LOGS=1)",
    )
    p.add_argument(
        "--per-seed-log-dir",
        type=str,
        default=str(DEFAULT_PER_SEED_LOG_DIR),
        help=f"Directory for per-seed log files (default {DEFAULT_PER_SEED_LOG_DIR}); used with --per-seed-logs or MYRL_PER_SEED_LOGS",
    )
    p.add_argument(
        "--parallel-seeds",
        action="store_true",
        help="Spawn one process per seed (DEFAULT_TEN_EVAL_SEEDS or --seeds); each writes only to logs_2/..._seedN.log (no master log). Mutually exclusive with --seed",
    )
    p.add_argument(
        "--log-to-file",
        type=str,
        default=None,
        help="Redirect stdout/stderr to this file (used by --parallel-seeds workers; rarely needed manually)",
    )
    # Logging/progress style (match test_SAC defaults).
    p.add_argument("--step-log-every", type=int, default=50, help="Print step log every N steps")
    p.add_argument(
        "--verbose-step-log",
        action="store_true",
        default=True,
        help="Enable step-level logs (default: on)",
    )
    p.add_argument(
        "--no-verbose-step-log",
        action="store_false",
        dest="verbose_step_log",
        help="Disable step-level logs",
    )
    p.add_argument(
        "--use-tqdm-progress",
        action="store_true",
        default=False,
        help="Show tqdm progress bar (recommended only when verbose logging is off)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.parallel_seeds and args.log_to_file:
        raise ValueError("--parallel-seeds 与 --log-to-file 不能同时用于同一进程（并行模式由子进程写日志）")

    if args.parallel_seeds:
        _run_parallel_seed_workers(args)
        return

    test_scenarios = int(args.test_scenarios)
    if args.episodes is not None:
        test_scenarios = int(args.episodes)

    if test_scenarios <= 0:
        raise ValueError("--test-scenarios must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    if args.log_to_file:
        _redirect_stdio_to_file(Path(args.log_to_file))

    env_per_seed = os.environ.get("MYRL_PER_SEED_LOGS", "").strip().lower() in ("1", "true", "yes")
    use_per_seed_logs = bool(args.per_seed_logs) or env_per_seed
    if args.log_to_file:
        use_per_seed_logs = False

    if args.seeds is not None:
        if args.seed is not None:
            print("[WARN] Both --seeds and --seed were set; using --seeds only.")
        seed_runs: list[Optional[int]] = _parse_seeds_csv(args.seeds)
    elif args.seed is not None:
        seed_runs = [int(args.seed)]
    elif use_per_seed_logs:
        seed_runs = list(DEFAULT_TEN_EVAL_SEEDS)
    else:
        # One run with legacy behavior: expose a reproducible state marker from numpy.
        seed_runs = [None]

    if args.disable_plotting is not None and args.enable_plotting is not None:
        # both flags present: last one wins in argparse ordering, but make it explicit.
        disable_plotting = bool(args.disable_plotting) and not bool(args.enable_plotting)
    elif args.enable_plotting is not None:
        disable_plotting = not bool(args.enable_plotting)
    else:
        disable_plotting = _auto_disable_plotting(args.disable_plotting)

    world_file = _resolve_world_file(args.world_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selection = _select_best_models()

    sim = SIM(world_file=str(world_file), disable_plotting=disable_plotting)

    # Instantiate system from the simulator's authoritative LiDAR metadata.
    models_dir = MODEL_ROOT
    system = HierarchicalNavigationSystem(
        lidar_metadata=sim.get_lidar_metadata(),
        action_dim=2,
        max_action=1.0,
        device=device,
        load_models=False,
        models_directory=models_dir,
    )

    # Load best models.
    system.high_level_planner.load_model(selection.high_prefix, selection.high_dir)
    _load_low_level_weights_map_location(system, selection.low_dir, selection.low_prefix, device)

    device_str = str(device)
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            device_str = f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})"
        except Exception:
            device_str = "cuda"

    # Align output style with robot_nav/test_SAC.py
    if args.seeds is not None:
        print(f"[Seed] Multi-seed mode: {seed_runs}")
    elif args.seed is not None:
        print(f"[Seed] Using random seed: {args.seed}")
    elif use_per_seed_logs:
        print(f"[Seed] Per-seed logs: dir={Path(args.per_seed_log_dir).resolve()} seeds={list(DEFAULT_TEN_EVAL_SEEDS)}")
    else:
        print(f"[Seed] No --seed/--seeds; numpy initial state marker: {int(np.random.get_state()[1][0])}")
    print("=" * 60)
    print("🧪 MYRL BEST-MODEL TEST CONFIG")
    print(f"   • Device:        {device_str}")
    print(f"   • World file:     {world_file}")
    print(f"   • Best root:      {_resolve_best_models_root()}")
    print(f"   • Config root:    {EXPLICIT_BEST_MODELS_ROOT}")
    print(f"   • Config epoch:   {EXPLICIT_MODEL_EPOCH if EXPLICIT_MODEL_EPOCH is not None else 'AUTO_LATEST'}")
    print(f"   • Config high:    {EXPLICIT_HIGH_LEVEL_PREFIX if EXPLICIT_HIGH_LEVEL_PREFIX is not None else 'AUTO_LATEST'}")
    print(f"   • Config low:     {EXPLICIT_LOW_LEVEL_PREFIX if EXPLICIT_LOW_LEVEL_PREFIX is not None else 'AUTO_LATEST'}")
    print(f"   • High-level:     {selection.high_prefix}  (dir: {selection.high_dir})")
    print(f"   • Low-level:      {selection.low_prefix}   (dir: {selection.low_dir})")
    print(f"   • Max steps/ep:   {args.max_steps}, Scenarios: {test_scenarios}")
    print(f"   • Plotting:       {'OFF' if disable_plotting else 'ON'}")
    print(f"   • Step log:       {'ON' if args.verbose_step_log else 'OFF'} (every {args.step_log_every} steps)")
    if args.use_tqdm_progress:
        print("   • Progress bar:   ON (tqdm)")
    else:
        print("   • Progress bar:   OFF")
    print("=" * 60)

    summaries: list[EvalRunSummary] = []
    seeds_for_agg: list[int] = []

    log_dir = Path(args.per_seed_log_dir)
    if use_per_seed_logs:
        log_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    world_stem = world_file.stem

    for run_i, seed_opt in enumerate(seed_runs):
        if seed_opt is None:
            seed_used = int(np.random.get_state()[1][0])
        else:
            _set_all_seeds(int(seed_opt))
            seed_used = int(seed_opt)

        seeds_for_agg.append(seed_used)

        seed_file = log_dir / f"test_{world_stem}_{run_ts}_seed{seed_used}.log"
        old_stdout = sys.stdout
        seed_fp = None
        if use_per_seed_logs:
            seed_fp = open(seed_file, "w", encoding="utf-8")
            sys.stdout = _TeeStream(old_stdout, seed_fp)

        try:
            print("..............................................")
            if len(seed_runs) > 1:
                print(f"Run {run_i + 1}/{len(seed_runs)}  |  seed={seed_used}")
            print(f"Testing {test_scenarios} scenarios  |  [Seed] {seed_used}")
            if use_per_seed_logs:
                print(f"[Per-seed log file] {seed_file.resolve()}")

            summary = _evaluate_scenarios(
                system,
                sim,
                test_scenarios=test_scenarios,
                max_steps=int(args.max_steps),
                verbose_step_log=bool(args.verbose_step_log),
                step_log_every=int(args.step_log_every),
                use_tqdm_progress=bool(args.use_tqdm_progress),
            )
            summaries.append(summary)
            _print_run_summary(summary, seed_used)
        finally:
            if use_per_seed_logs and seed_fp is not None:
                sys.stdout.flush()
                seed_fp.flush()
                sys.stdout = old_stdout
                seed_fp.close()

    _print_multi_seed_aggregate(summaries, seeds_for_agg)


if __name__ == "__main__":
    main()
