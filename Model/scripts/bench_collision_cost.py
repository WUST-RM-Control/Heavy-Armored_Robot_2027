"""量碰撞开销：球体 vs 凸包（vs 胶囊）到底差多少、差在哪一层。

回答两个常见问题：

1. **「两套碰撞体的仿真计算量差不多吗？」** —— 整机对比（第 ②③ 节）：
   同样的初值、同样的控制，逐项拆出 broad phase / narrow phase / 求解器 / 动力学。
2. **「做碰撞检测谁更合适？谁适合先粗检查？」** —— 图元级对比（第 ① 节）：
   造一批完全重叠的合成场景，量「每个接触多少纳秒」，再看网格顶点数的缩放。

第 ② 节顺带复刻 `test.py` 的场景（5000 步后换目标），并且**不开 viewer**：
viewer 的 `sync()` 会把物理循环压到渲染帧率上（实测 5000 步 ≈ 52s ≈ 96 步/秒，
而纯物理只要 0.24~0.53s），所以「跑多久」根本不是仿真开销的指标。

用法::

    python scripts/bench_collision_cost.py                 # 全部
    python scripts/bench_collision_cost.py --skip-synthetic
    python scripts/bench_collision_cost.py --steps 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

VARIANTS = ("convex", "sphere", "capsule")
TIMERS = ("POSITION", "POS_COLLISION", "COL_BROAD", "COL_NARROW", "CONSTRAINT")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def snapshot(data) -> dict:
    import mujoco

    return {name: data.timer[getattr(mujoco.mjtTimer, f"mjTIMER_{name}")].duration
            for name in TIMERS}


def delta(before: dict, after: dict, count: int, scale: float = 1e3) -> dict:
    return {name: (after[name] - before[name]) / count * scale for name in TIMERS}


def load(variant: str, collision: bool = True):
    """载入某个变体的 MJCF；collision=False 时把所有 geom 的碰撞位关掉"""
    import mujoco

    path = config.mjcf_path(variant, "motor")
    model = mujoco.MjModel.from_xml_path(str(path))
    if not collision:
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    data.qpos[:] = [0, 0, 0, 0, 0, 3.14, 0]
    data.ctrl[:] = 0
    return model, data


def make_controller(model, data, root: Path):
    """优先用工程里编译好的 CascadePID（和 test.py 一致），没有就退回内置 PD"""
    try:
        sys.path.insert(0, str(root / "build"))
        from control_arm_lib import CascadePID  # noqa: PLC0415

        gains = [(4, 8), (20, 40), (10, 20), (4, 8), (4, 8), (3, 6), (2, 4)]
        pids = [CascadePID(2, 0, 0, 0, 0, 3.14, 1, 0.1, 0, 0, kp, kd) for kp, kd in gains]

        def control(target):
            for index, pid in enumerate(pids):
                pid.PID_Position_Calculate(target[index], data.qpos[index],
                                           data.qvel[index], model.opt.timestep)
                data.ctrl[index] = pid.Get_Inner_Output()
        return control, "CascadePID（build/control_arm_lib）"
    except Exception as exc:                      # 没有编译产物也能跑
        print(f"  NOTE: 用不了 CascadePID（{exc}），退回内置 PD")

        def control(target):
            data.ctrl[:] = 400.0 * (np.asarray(target) - data.qpos) - 40.0 * data.qvel
        return control, "内置 PD"


# --------------------------------------------------------------------------- #
# ① 图元级：每个接触多少纳秒
# --------------------------------------------------------------------------- #
def synthetic_scene(geom_template: str, count: int, meshes: dict[str, str],
                    spacing: float = 0.02) -> str:
    def block(prefix: str) -> str:
        side = int(np.ceil(np.sqrt(count)))
        return "\n".join(
            geom_template.format(i=f"{prefix}{index}", x=(index % side) * spacing,
                                 y=(index // side) * spacing, z=0.0)
            for index in range(count))

    assets = "".join(f'<mesh name="{name}" file="{path}"/>' for name, path in meshes.items())
    return f"""
<mujoco>
  <option gravity="0 0 0" timestep="0.002"/>
  <asset>{assets}</asset>
  <worldbody>
    <body name="a"><freejoint/>{block("a")}</body>
    <body name="b"><freejoint/>{block("b")}</body>
  </worldbody>
</mujoco>
"""


def synthetic_case(geom_template: str, count: int, meshes: dict[str, str],
                   repeats: int = 1500) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_string(synthetic_scene(geom_template, count, meshes))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = snapshot(data)
    started = time.perf_counter()
    for _ in range(repeats):
        mujoco.mj_forward(model, data)
    wall = (time.perf_counter() - started) / repeats
    timer = delta(before, snapshot(data), repeats, scale=1e6)
    return dict(ngeom=model.ngeom, ncon=data.ncon, us=wall * 1e6, timer=timer)


def bench_synthetic(count: int = 64) -> None:
    import shutil

    import trimesh

    work = config.MODEL_DIR / ".bench_meshes"
    work.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for name, mesh in (("box8", trimesh.creation.box(extents=[0.1, 0.1, 0.1])),
                       ("ico42", trimesh.creation.icosphere(subdivisions=1, radius=0.05)),
                       ("ico162", trimesh.creation.icosphere(subdivisions=2, radius=0.05)),
                       ("ico642", trimesh.creation.icosphere(subdivisions=3, radius=0.05))):
        path = work / f"{name}.obj"
        mesh.export(path)
        meshes[name] = str(path)        # 绝对路径：from_xml_string 没有基准目录
    cases = [
        ("球 r=50mm", '<geom name="{i}" type="sphere" size="0.05" pos="{x} {y} {z}"/>'),
        ("网格 box 8 顶点", '<geom name="{i}" type="mesh" mesh="box8" pos="{x} {y} {z}"/>'),
        ("网格 ico 42 顶点", '<geom name="{i}" type="mesh" mesh="ico42" pos="{x} {y} {z}"/>'),
        ("网格 ico 162 顶点", '<geom name="{i}" type="mesh" mesh="ico162" pos="{x} {y} {z}"/>'),
        ("网格 ico 642 顶点", '<geom name="{i}" type="mesh" mesh="ico642" pos="{x} {y} {z}"/>'),
    ]
    print("=" * 92)
    print(f"① 图元级窄相位成本：两侧各 {count} 个 geom 完全重叠（{count}×{count} 对候选）")
    print("=" * 92)
    print(f"{'图元':20s} {'geom':>5s} {'ncon':>6s} {'broad μs':>9s} {'narrow μs':>10s} "
          f"{'每个接触 ns':>12s}")
    for label, template in cases:
        try:
            result = synthetic_case(template, count, meshes)
        except Exception as exc:
            print(f"{label:20s} 失败（{str(exc).splitlines()[0][:50]}）")
            continue
        per = result["timer"]["COL_NARROW"] / result["ncon"] * 1000 if result["ncon"] else np.nan
        print(f"{label:20s} {result['ngeom']:5d} {result['ncon']:6d} "
              f"{result['timer']['COL_BROAD']:9.1f} {result['timer']['COL_NARROW']:10.1f} "
              f"{per:12.1f}")
    print("  （broad phase 是 AABB 级的，跟图元类型无关；narrow 才是图元成本，")
    print("    网格随顶点数近似线性：42→642 顶点，每个接触 1.2μs → 3.5μs）")
    shutil.rmtree(work, ignore_errors=True)     # 临时网格用完就删


# --------------------------------------------------------------------------- #
# ②③ 整机级：test.py 的场景，拆开每一层
# --------------------------------------------------------------------------- #
def bench_robot(steps: int) -> dict:
    import mujoco

    print()
    print("=" * 92)
    print(f"② 整机 {steps} 步（复刻 test.py：qpos[5]=3.14，一半处换目标，不开 viewer）")
    print("=" * 92)
    print(f"{'模型':9s} {'总耗时s':>8s} {'ms/步':>7s} {'碰撞ms':>8s} {'├broad':>8s} "
          f"{'└narrow':>9s} {'求解ms':>8s} {'动力学ms':>9s} {'平均ncon':>9s} {'最大':>6s}")
    results = {}
    for variant in VARIANTS:
        model, data = load(variant)
        control, source = make_controller(model, data, config.MODEL_DIR.parent)
        target = [0, 0, 0, 0, 0, 3.14, 0]
        for _ in range(200):                      # 预热
            mujoco.mj_step(model, data)
            control(target)
        before = snapshot(data)
        ncon_sum = ncon_max = 0
        started = time.perf_counter()
        for step in range(steps):
            mujoco.mj_step(model, data)
            control(target)
            if step == steps // 2:
                target = [1, 0, 0, 0, 0, 3.14, 1]
            ncon_sum += data.ncon
            ncon_max = max(ncon_max, data.ncon)
        wall = time.perf_counter() - started
        timer = delta(before, snapshot(data), steps)
        dynamics = timer["POSITION"] - timer["POS_COLLISION"]
        results[variant] = dict(wall=wall)
        print(f"{variant:9s} {wall:8.2f} {wall / steps * 1e3:7.3f} "
              f"{timer['POS_COLLISION']:8.3f} {timer['COL_BROAD']:8.3f} "
              f"{timer['COL_NARROW']:9.3f} {timer['CONSTRAINT']:8.3f} {dynamics:9.3f} "
              f"{ncon_sum / steps:9.2f} {ncon_max:6d}")
    print(f"  （控制器：{source}）")

    print()
    print("③ 关掉全部碰撞（纯动力学地板）：看碰撞+求解占总时间的比例")
    print("=" * 92)
    for variant in VARIANTS:
        model, data = load(variant, collision=False)
        control, _ = make_controller(model, data, config.MODEL_DIR.parent)
        target = [0, 0, 0, 0, 0, 3.14, 0]
        for _ in range(200):
            mujoco.mj_step(model, data)
            control(target)
        started = time.perf_counter()
        for step in range(steps):
            mujoco.mj_step(model, data)
            control(target)
            if step == steps // 2:
                target = [1, 0, 0, 0, 0, 3.14, 1]
        wall = time.perf_counter() - started
        full = results[variant]["wall"]
        print(f"{variant:9s} 无碰撞 {wall / steps * 1e3:7.3f} ms/步   "
              f"有碰撞 {full / steps * 1e3:7.3f} ms/步   "
              f"→ 碰撞+求解占 {(1 - wall / full) * 100:5.1f}%")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="碰撞开销基准：球体 vs 凸包 vs 胶囊。")
    parser.add_argument("--steps", type=int, default=5000, help="整机对比的步数")
    parser.add_argument("--count", type=int, default=64, help="合成场景每侧 geom 数")
    parser.add_argument("--skip-synthetic", action="store_true", help="跳过图元级对比")
    parser.add_argument("--skip-robot", action="store_true", help="跳过整机对比")
    args = parser.parse_args()

    if not args.skip_synthetic:
        bench_synthetic(args.count)
    if not args.skip_robot:
        bench_robot(args.steps)

    print()
    print("=" * 92)
    print("怎么用这些数：")
    print("  * 想看「谁仿真更贵」，看整机表的 ms/步 —— 但注意 viewer 的 sync() 会把循环")
    print("    压到渲染帧率（~96 步/秒），『跑完 5000 步用了多久』量的是渲染不是物理；")
    print("  * 想看「谁适合粗检查」，看图元级表：球是解析距离、每接触最便宜、且与网格")
    print("    分辨率无关，天生适合 broad phase / 距离剪枝；凸包每个接触贵 3~9 倍，")
    print("    但接触点少、法向准，适合 narrow phase。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
