"""OBB 计算：用 Open3D 求每个连杆的有向包围盒，并导出成盒网格。

流程里的第 [3] 步，输入同凸分解（简化后的 STL）。

输出::

    meshs/obb/LINK1_obb.obj        # 8 顶点 12 三角面的盒子，在连杆坐标系里
    meshs/obb/obb_params.json      # center / extent / 旋转矩阵 / 四元数，便于对照

为什么要导出成 obj 而不是直接用 URDF 的 ``<box>`` 图元：这样 ``_obb.urdf`` 的
碰撞词条和 ``_convex.urdf`` 完全同构（都是 mesh），两条流水线可以逐行对照；
MuJoCo 拿到盒网格后取的凸包就是长方体本身，碰撞结果与 ``<box>`` 图元一致。
(json 里留了 center/extent/四元数，想换成 ``<box>`` 图元直接取用即可。)

用法::

    python scripts/decompose_obb.py
    python scripts/decompose_obb.py --only LINK4 --margin 0.001
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def obb_of(path: Path) -> tuple[o3d.geometry.OrientedBoundingBox, int]:
    """读网格并求 OBB，返回 (obb, 原始面数)"""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Open3D 读不出网格顶点：{path}")
    faces = len(mesh.triangles) if mesh.triangles is not None else 0
    return mesh.get_oriented_bounding_box(), faces


def box_mesh(obb: o3d.geometry.OrientedBoundingBox, margin: float) -> o3d.geometry.TriangleMesh:
    """把 OBB 变成一个居中的盒网格（每边外扩 margin）"""
    extent = np.asarray(obb.extent, dtype=float) + 2.0 * margin
    box = o3d.geometry.TriangleMesh.create_box(*extent)
    box.translate(-extent / 2.0)                    # create_box 的角点在原点，挪成居中
    box.rotate(np.asarray(obb.R, dtype=float), center=(0.0, 0.0, 0.0))
    box.translate(np.asarray(obb.center, dtype=float))
    box.compute_vertex_normals()
    return box


def quaternion_from_matrix(rotation: np.ndarray) -> list[float]:
    """旋转矩阵 → 四元数 (w, x, y, z)，MuJoCo 的 quat 顺序"""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    w, x, y, z = trimesh.transformations.quaternion_from_matrix(transform)
    return [float(w), float(x), float(y), float(z)]


def obb_one(stem: str, margin: float = config.OBB_MARGIN, force: bool = False) -> dict:
    """算一个连杆的 OBB，写出 obj，返回参数字典"""
    source = config.visual_mesh(stem)
    target = config.obb_path(stem)
    if target.is_file() and not force:
        print(f"  {stem:10s} 已有 {target.name}，跳过（--force 可重算）")
        with_meta = target.with_suffix(".json")
        return json.loads(with_meta.read_text()) if with_meta.is_file() else {"stem": stem}

    obb, faces = obb_of(source)
    started = time.time()
    config.OBB_DIR.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(target), box_mesh(obb, margin))

    center = np.asarray(obb.center, dtype=float)
    extent = np.asarray(obb.extent, dtype=float)
    rotation = np.asarray(obb.R, dtype=float)

    # 贴合度：OBB 体积 vs 网格凸包体积（1.0 = 零件本身就是个盒子）
    mesh = trimesh.load(str(source), force="mesh")
    hull_volume = float(abs(mesh.convex_hull.volume)) or float("nan")
    obb_volume = float(np.prod(extent))

    info = dict(
        stem=stem,
        source=source.name,
        faces=faces,
        center=[float(v) for v in center],
        extent=[float(v) for v in extent],
        rotation=[[float(v) for v in row] for row in rotation],
        quat=quaternion_from_matrix(rotation),
        obb_volume=obb_volume,
        hull_volume=hull_volume,
        fill_ratio=float(hull_volume / obb_volume) if obb_volume else float("nan"),
        margin=margin,
        file=target.name,
        seconds=time.time() - started,
    )
    print(f"  {stem:10s} {source.name:24s} → OBB {np.round(extent, 4)} "
          f"体积比 {info['fill_ratio']:.3f} ({time.time() - started:.1f}s)")
    return info


def obb_all(only: list[str] | None = None, margin: float = config.OBB_MARGIN,
            force: bool = False) -> dict[str, dict]:
    """批量算 OBB，返回 ``{stem: 参数}``，并写出 obb_params.json"""
    config.ensure_dirs()
    stems = discover(only)
    if not stems:
        print(f"WARNING: {config.VISUAL_DIR} 下没有网格可算 OBB")
        return {}

    print(f"OBB 计算：{len(stems)} 个网格（Open3D），外扩余量 {margin * 1000:.2f}mm")
    report = {stem: obb_one(stem, margin=margin, force=force) for stem in stems}

    if config.OBB_WRITE_PARAMS:
        params_file = config.OBB_DIR / config.OBB_PARAMS_FILE
        params_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"参数已写入 {params_file.relative_to(config.MODEL_DIR)}")

    print(f"完成：{len(report)} 个 OBB，输出到 {config.OBB_DIR}")
    return report


def discover(only: list[str] | None = None) -> list[str]:
    """要算 OBB 的连杆 stem（和凸分解一致，去重排序）"""
    stems = []
    for path in sorted(config.VISUAL_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in (".stl", ".obj", ".ply"):
            stem = config.clean_stem(path.stem)
            if config.is_simplified(path) or stem in stems:
                continue
            if only and stem not in only:
                continue
            stems.append(stem)
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(description="用 Open3D 计算连杆 OBB 并导出盒网格。")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些连杆")
    parser.add_argument("--margin", type=float, default=config.OBB_MARGIN,
                        help=f"每边外扩余量（米，默认 {config.OBB_MARGIN}）")
    parser.add_argument("--force", action="store_true", help="已有结果也重算")
    args = parser.parse_args()

    obb_all(only=args.only, margin=args.margin, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
