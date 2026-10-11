"""碰撞体质量诊断：真实网格 / 凸包 / 球体（foam） / 胶囊 / OBB 对比。

流程之外的独立分析工具（不改任何产物），用来回答两个问题：

    **① 这套碰撞体「胖」了多少？**（胖 → 幻影接触）
    **② 这套碰撞体「漏」了多少？**（漏 → 真接触被漏掉）

做法是逐连杆、逐对比较「碰撞体之间的间隙」和「真实网格之间的间隙」：

* 真实网格在零位（qpos=0）几乎不接触时，碰撞体却互相接触 → **幻影接触**；
* 幻影接触会让 MuJoCo 在关节零位就把连杆顶开，PD 控制跟不到位、
  力矩控制的响应也会被接触反力改写；
* 反过来，如果网格表面有点没被任何球盖住（「漏」> 0），那片几何就是碰撞盲区。

真值取自视觉网格（顶点 + 面心，KDTree 最近邻），碰撞体取 MuJoCo 载入后的 geom
采样点。输出四张表：

1. 真实网格间隙（非相邻连杆对，从最小开始）；
2. 逐连杆贴近度：球（或胶囊）表面到本零件表面的距离统计（中位 / 95% / 最大）
   ——「胖出来多少」；同时给出「网格表面到球并集的最大间隙」——「漏多少」。
   两个数都小于该连杆周围的最小真实间隙，这套近似才既不会假接触也不会漏接触；
3. 逐对间隙：真实网格 vs 碰撞体（球体侧 < ``--max-gap`` 的对）；
4. 各变体在 qpos=0 的接触数与最深穿透。

用法::

    python scripts/analyze_collision.py                     # 默认分析 sphere
    python scripts/analyze_collision.py --primitive capsule  # 老胶囊方案对照
    python scripts/analyze_collision.py --max-gap 20 --samples 40000
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def collision_gap(path: Path, robot: str) -> dict:
    """各变体在 qpos=0 的接触统计（接触对 + 最深穿透）"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    deepest, pairs = 0.0, []
    for index in range(data.ncon):
        contact = data.contact[index]
        deepest = min(deepest, float(contact.dist))
        names = []
        for geom in contact.geom:
            body = int(model.geom_bodyid[geom])
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world")
        pairs.append(f"{names[0]}↔{names[1]}")
    return dict(model=model, ncon=int(data.ncon), deepest=abs(deepest), pairs=sorted(set(pairs)))


def mesh_samples(stem: str, limit: int, seed: int = 0) -> np.ndarray:
    """视觉网格的表面采样点（顶点 + 三角面心，超量就随机抽）"""
    mesh = trimesh.load(str(config.visual_mesh(stem)), force="mesh", process=False)
    points = np.vstack([np.asarray(mesh.vertices, dtype=float),
                        np.asarray(mesh.triangles, dtype=float).mean(axis=1)])
    if len(points) > limit:
        points = points[np.random.default_rng(seed).choice(len(points), limit, replace=False)]
    return points


def primitives(model, kind: str) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """列出某个变体里指定类型的 geom：``[(stem, geom_id, center, size), ...]``"""
    import mujoco

    wanted = {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
              "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}[kind]
    found = []
    for index in range(model.ngeom):
        if model.geom_type[index] != wanted:
            continue
        body = int(model.geom_bodyid[index])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world"
        geom = model.geom(index)
        found.append(("base_link" if name == "world" else name, index,
                      np.asarray(geom.pos, dtype=float), np.asarray(geom.size, dtype=float)))
    return found


def sphere_surface_samples(size: np.ndarray, count: int = 600, seed: int = 0) -> np.ndarray:
    """球面采样点（局部坐标）"""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions * float(size[0])


def capsule_surface_samples(size: np.ndarray, count: int = 6000, seed: int = 0) -> np.ndarray:
    """胶囊表面的采样点（局部坐标，轴 = +Z）"""
    rng = np.random.default_rng(seed)
    radius, half = float(size[0]), float(size[1])
    # 柱面
    count_body = count // 2
    z = rng.uniform(-half, half, count_body)
    angle = rng.uniform(0.0, 2.0 * np.pi, count_body)
    body = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])
    # 两个球头
    count_cap = (count - count_body) // 2
    theta = np.arccos(rng.uniform(0.0, 1.0, count_cap))
    angle = rng.uniform(0.0, 2.0 * np.pi, count_cap)
    caps = []
    for sign in (-1.0, 1.0):
        caps.append(np.column_stack([
            radius * np.sin(theta) * np.cos(angle),
            radius * np.sin(theta) * np.sin(angle),
            sign * (half + radius * np.cos(theta)),
        ]))
    return np.vstack([body, *caps])


def primitive_surface_points(model, data, kind: str, per_geom: int, seed: int = 0
                             ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """各连杆的碰撞体表面采样点（世界坐标）→ ``{stem: (点, 球心, 半径)}``"""
    import mujoco

    grouped: dict[str, list] = {}
    for stem, index, center, size in primitives(model, kind):
        grouped.setdefault(stem, []).append((index, center, size))

    result = {}
    for stem, items in grouped.items():
        body = items[0][0]
        body_id = int(model.geom_bodyid[body])
        rotation = np.zeros(9)
        points, centers, radii = [], [], []
        for index, center, size in items:
            geom = model.geom(index)
            mujoco.mju_quat2Mat(rotation, geom.quat)
            local = (sphere_surface_samples(size, per_geom, seed)
                     if kind == "sphere"
                     else capsule_surface_samples(size, per_geom, seed))
            points.append(local @ rotation.reshape(3, 3).T + center)
            centers.append(center)
            radii.append(float(size[0]))
        result[stem] = (np.vstack(points) + data.xpos[body_id],
                        np.asarray(centers) + data.xpos[body_id],
                        np.asarray(radii))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="碰撞体质量诊断（真实网格 vs 凸包 vs 球体 vs 胶囊 vs OBB）。")
    parser.add_argument("--robot", default=config.DEFAULT_ROBOT)
    parser.add_argument("--primitive", choices=("sphere", "capsule"), default="sphere",
                        help="②③ 两张表分析哪种碰撞体（默认 sphere=foam 球体）")
    parser.add_argument("--samples", type=int, default=60000,
                        help="每个网格的表面采样上限（默认 60000）")
    parser.add_argument("--max-gap", type=float, default=20.0,
                        help="真实间隙表只列出小于这个值（毫米）的连杆对")
    args = parser.parse_args()

    import mujoco

    stems = sorted((config.clean_stem(p.stem) for p in config.VISUAL_DIR.iterdir()
                    if p.is_file() and not config.is_simplified(p)), key=str.lower)

    # ---- 1. 真实网格间隙（非相邻连杆对） -------------------------------- #
    print("=" * 78)
    print("① 真实视觉网格在 qpos=0 的最小间隙（相邻连杆对已按 URDF 语义排除）")
    print("=" * 78)
    model = mujoco.MjModel.from_xml_path(str(config.mjcf_path("convex", "pd", args.robot)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    world = {}
    body_ids = {}
    for stem in stems:
        body = 0 if stem == "base_link" else mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, stem)
        body_ids[stem] = body
        points = mesh_samples(stem, args.samples) + data.xpos[body]
        world[stem] = (points, cKDTree(points))

    def adjacent(left: str, right: str) -> bool:
        """URDF 语义上的相邻连杆（父子）——它们本来就不该互撞，不参与比较"""
        return (int(model.body_parentid[body_ids[right]]) == body_ids[left]
                or int(model.body_parentid[body_ids[left]]) == body_ids[right])

    rows = []
    for left, right in itertools.combinations(stems, 2):
        if adjacent(left, right):
            continue
        gap = max(float(world[right][1].query(world[left][0])[0].min()),
                  float(world[left][1].query(world[right][0])[0].min()))
        rows.append((gap, left, right))
    rows.sort()
    print(f"  最小真实间隙：{rows[0][0] * 1000:.2f}mm（{rows[0][1]} ↔ {rows[0][2]}）")
    print(f"  次小真实间隙：{rows[1][0] * 1000:.2f}mm（{rows[1][1]} ↔ {rows[1][2]}）")
    for gap, left, right in rows[:8]:
        print(f"    {left:10s} ↔ {right:10s} {gap * 1000:9.2f}mm")
    print(f"  （下表只列 < {args.max_gap:.0f}mm 的，其余按「离得远」处理）")

    # ---- 2. 碰撞体胖/漏多少 --------------------------------------------- #
    kind = args.primitive
    label = "球体" if kind == "sphere" else "胶囊"
    path = config.mjcf_path(kind, "pd", args.robot)
    if not path.is_file():
        print(f"\n  没有 {path.name}，跳过 ②③（先跑 run_pipeline.py）")
        return 0
    print()
    print("=" * 78)
    print(f"② {label}贴不贴合：{label}表面→本零件表面（胖），本零件表面→{label}（漏）")
    print("=" * 78)
    primitive_model = mujoco.MjModel.from_xml_path(str(path))
    primitive_data = mujoco.MjData(primitive_model)
    mujoco.mj_forward(primitive_model, primitive_data)
    surfaces = primitive_surface_points(primitive_model, primitive_data, kind,
                                        per_geom=600 if kind == "sphere" else 6000)

    print(f"  {'link':10s} {'geom':>4s} {'半径mm':>14s} {'胖 中位':>8s} {'胖 P95':>8s} "
          f"{'胖 最大':>9s} {'漏 最大':>9s}   周围真实间隙")
    for stem in stems:
        if stem not in surfaces:
            continue
        points, centers, radii = surfaces[stem]
        tree = cKDTree(world[stem][0])
        fat = tree.query(points)[0]
        # 漏：网格表面点到球并集的间隙（只有球体才有这个指标；胶囊同理可算）
        k = int(min(16, len(radii)))
        if kind == "sphere":
            distance, index = cKDTree(centers).query(world[stem][0], k=k)
            leak = float((distance - radii[index]).min(axis=1).max())
        else:
            leak = float("nan")
        nearest = min((gap for gap, a, b in rows if stem in (a, b)), default=float("inf"))
        print(f"  {stem:10s} {len(radii):4d} {radii.min() * 1000:6.1f}~{radii.max() * 1000:6.1f} "
              f"{np.median(fat) * 1000:7.2f}mm {np.percentile(fat, 95) * 1000:7.2f}mm "
              f"{fat.max() * 1000:8.2f}mm {leak * 1000:8.2f}mm   {nearest * 1000:7.2f}mm")
    print("  （「胖 最大」大于周围真实间隙 → 一定会假接触；"
          "「漏 最大」大于 0 → 那片几何是碰撞盲区）")

    # ---- 3. 逐对间隙：真实网格 vs 碰撞体 -------------------------------- #
    print()
    print("=" * 78)
    print(f"③ 逐对间隙对比：真实网格 与 {label}（qpos=0，只列{label}侧 < "
          f"{args.max_gap:.0f}mm 的对）")
    print("=" * 78)
    trees = {stem: cKDTree(points) for stem, (points, _, _) in surfaces.items()}
    print(f"  {'连杆对':24s} {'真实间隙':>10s} {label + '间隙':>10s} {'误差':>10s}  判定")
    for gap, left, right in rows:
        if left not in trees or right not in trees:
            continue
        gap_primitive = max(
            float(trees[right].query(surfaces[left][0])[0].min()),
            float(trees[left].query(surfaces[right][0])[0].min()))
        if gap_primitive > args.max_gap / 1000.0:
            continue
        if gap < 0.001:                                   # 真实网格本来就贴着/插着
            verdict = "真实就接触，两者都报——正常"
        elif gap_primitive < gap * 0.5:
            verdict = "幻影接触 ✗"
        else:
            verdict = "偏保守"
        print(f"  {left + ' ↔ ' + right:24s} {gap * 1000:8.2f}mm {gap_primitive * 1000:8.2f}mm "
              f"{(gap_primitive - gap) * 1000:+8.2f}mm  {verdict}")

    # ---- 4. 各变体零位接触 ---------------------------------------------- #
    print()
    print("=" * 78)
    print("④ 各碰撞体变体在 qpos=0 的接触（幻影接触一眼可见）")
    print("=" * 78)
    for variant in ("convex", "sphere", "capsule", "obb"):
        variant_path = config.mjcf_path(variant, "pd", args.robot)
        if not variant_path.is_file():
            continue
        info = collision_gap(variant_path, args.robot)
        pairs = "、".join(info["pairs"]) if info["pairs"] else "无接触"
        print(f"  {variant:8s} ngeom={info['model'].ngeom:4d} ncon={info['ncon']:3d} "
              f"最深穿透 {info['deepest'] * 1000:8.2f}mm  {pairs}")

    print("=" * 78)
    print("怎么读这份报告：「胖 最大」大于周围真实间隙 → 假接触；「漏 最大」大于 0 → 盲区。")
    print("球体方案想再紧一点：调大 decompose_sphere.py 的 --depth/--branch，")
    print("或者把个别连杆塞进 config.SPHERE_SKIP_LINKS 退回凸包。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
