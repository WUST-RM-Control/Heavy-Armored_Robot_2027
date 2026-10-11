"""STL 简化：把 ``meshs/visual/`` 下的网格压到 MuJoCo 能接受的面数。

流程里的第 [1] 步，凸分解 / 球分解 / OBB 都用它的输出。

规则
----
* 面数 ≤ ``MAX_MESH_FACES``（默认 10 万）：原样保留，URDF 视觉网格直接指向原件；
* 面数超标：用二次误差简化（QEM）压到上限，输出到
  ``meshs/simplified/<stem>_simplified.STL``；
* 原始 STL 永远不动，简化结果是**另存**的副本。

简化后端（按顺序自动选第一个可用的，失败会继续降级）
----------------------------------------------------
============ ==================================================== ==========
后端         来源                                                 质量
============ ==================================================== ==========
``fast``     ``fast_simplification``（trimesh 自带封装，C 实现）    最好
``open3d``   ``open3d.geometry.TriangleMesh`` 的 QEM              好
``chull``    ``trimesh.convex.convex_hull``（只用 numpy/scipy）    形状失真
``cluster``  ``trimesh.Trimesh.simplify_vertex_clustering``       形状失真
============ ==================================================== ==========

``fast_simplification`` 是 trimesh 的**可选**依赖（两个环境都如此），所以这里不能
硬依赖它：``~/mujoco_env`` 装了 0.1.9，而 ``.pixi/envs/default`` 的 trimesh 5.1
没装，从 pixi 解释器起流程就会 ``ModuleNotFoundError``。降级链保证任何装了
trimesh 的解释器都能跑完，实际用的是哪个后端会打印并记录在报告里。

用法::

    python scripts/simplify_mesh.py                  # 全部网格
    python scripts/simplify_mesh.py --only LINK4     # 只处理一个
    python scripts/simplify_mesh.py --max-faces 50000 --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

#: 后端名 → (中文说明, 质量分)。质量分只用于报告里的排序提示，不是精度保证
BACKENDS: dict[str, tuple[str, int]] = {
    "fast": ("fast_simplification（trimesh 自带封装）", 3),
    "open3d": ("open3d 的二次误差简化", 2),
    "chull": ("凸包（无简化库时的兜底）", 1),
    "cluster": ("顶点聚类（最后的兜底）", 0),
}


def _decimate_fast(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """QEM：fast_simplification（trimesh ``simplify_quadric_decimation`` 的底层）。

    这是 trimesh 的 extra（``recommend``）依赖，缺了就抛 ImportError 让上层降级。
    """
    from fast_simplification import simplify  # noqa: PLC0415
    vertices, faces = simplify(
        points=np.asarray(mesh.vertices, dtype=np.float64),
        triangles=np.asarray(mesh.faces, dtype=np.int32),
        target_count=int(max_faces),
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _decimate_open3d(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """QEM：open3d（两个环境都装了，是缺失 fast_simplification 时的主力后端）"""
    import open3d as o3d  # noqa: PLC0415

    omesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    simplified = omesh.simplify_quadric_decimation(
        target_number_of_triangles=int(max_faces))
    return trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles),
        process=False,
    )


def _decimate_chull(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """只依赖 numpy/scipy 的兜底：凸包（形状会失真，只保证不报错）"""
    return mesh.convex_hull


def _decimate_cluster(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """最后兜底：顶点聚类，体素边长按 bbox 对角线二分逼近目标面数"""
    pitch = float(np.linalg.norm(mesh.extents))
    for _ in range(60):
        simplified = mesh.simplify_vertex_clustering(pitch / 100.0)
        count = len(simplified.faces)
        if count <= max_faces and count > 0:
            return simplified
        pitch *= 1.15 if count > max_faces else 0.9
    return mesh.simplify_vertex_clustering(pitch / 100.0)


#: 降级链：按顺序尝试，第一个成功的就是结果
CASCADE = (("fast", _decimate_fast),
           ("open3d", _decimate_open3d),
           ("chull", _decimate_chull),
           ("cluster", _decimate_cluster))


def _check_backends() -> tuple[str | None, list[str]]:
    """探测首选后端 + 列出所有可用后端（只 import，不做简化）"""
    usable: list[str] = []
    for name in ("fast", "open3d"):
        try:
            __import__("fast_simplification" if name == "fast" else "open3d")
            usable.append(name)
        except ImportError:
            pass
    usable.extend(["chull", "cluster"])
    return (usable[0] if usable else None), usable


def _volume(mesh: trimesh.Trimesh) -> float:
    """体积；网格不封闭时退回凸包体积（凸包几乎总是封闭的）"""
    try:
        volume = float(abs(mesh.volume))
        if volume > 0.0:
            return volume
    except Exception:  # noqa: BLE001 - 退化网格的 volume 可能是 nan
        pass
    try:
        return float(abs(mesh.convex_hull.volume))
    except Exception:  # noqa: BLE001
        return 0.0


def _decimate(mesh: trimesh.Trimesh, max_faces: int) -> tuple[trimesh.Trimesh, str]:
    """按降级链简化，返回 ``(网格, 实际用的后端名)``"""
    errors: list[str] = []
    for name, func in CASCADE:
        try:
            simplified = func(mesh, max_faces)
        except Exception as exc:  # noqa: BLE001 - 任何后端失败都继续降级
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if simplified is None or len(simplified.faces) == 0:
            errors.append(f"{name}: 输出 0 面")
            continue
        return simplified, name
    raise RuntimeError("所有简化后端都失败：\n  " + "\n  ".join(errors))


def face_count(path: Path) -> int:
    """只读面数，用来快速判断要不要简化（trimesh 会做完整加载）"""
    return len(trimesh.load(str(path), force="mesh").faces)


def simplify_one(path: Path, max_faces: int = config.MAX_MESH_FACES,
                 force: bool = False) -> dict:
    """简化一个网格。

    返回 ``dict(stem, source, output, faces_before, faces_after, simplified,
    backend, volume_ratio, seconds)``；
    ``output`` 是 URDF 视觉网格最终该指向的文件（原件或简化副本）。
    """
    stem = config.clean_stem(path.stem)
    target = config.simplified_path(stem)
    started = time.time()

    mesh = trimesh.load(str(path), force="mesh")
    before = len(mesh.faces)

    if before <= max_faces:
        # 不需要简化：顺手清掉可能残留的旧简化副本，免得视觉网格指到过期文件
        if target.is_file():
            target.unlink()
            print(f"  {stem:10s} {before:7d} 面 ≤ {max_faces}，删除过期的 {target.name}")
        else:
            print(f"  {stem:10s} {before:7d} 面 ≤ {max_faces}，用原件")
        return dict(stem=stem, source=path, output=path, faces_before=before,
                    faces_after=before, simplified=False, backend="none",
                    volume_ratio=1.0, seconds=time.time() - started)

    if target.is_file() and not force:
        cached = len(trimesh.load(str(target), force="mesh").faces)
        print(f"  {stem:10s} {before:7d} 面 > {max_faces}，已有简化副本（{cached} 面），跳过")
        return dict(stem=stem, source=path, output=target, faces_before=before,
                    faces_after=cached, simplified=True, backend="cached",
                    volume_ratio=float("nan"), seconds=time.time() - started)

    simplified, backend = _decimate(mesh, max_faces)
    target.parent.mkdir(parents=True, exist_ok=True)
    simplified.export(str(target))
    after = len(simplified.faces)
    ratio = _volume(simplified) / _volume(mesh) if before else float("nan")
    print(f"  {stem:10s} {before:7d} 面 > {max_faces} → 简化到 {after} 面 "
          f"({target.name}, {BACKENDS[backend][0]}, 体积比 {ratio:.3f}, "
          f"{time.time() - started:.1f}s)")
    return dict(stem=stem, source=path, output=target, faces_before=before,
                faces_after=after, simplified=True, backend=backend,
                volume_ratio=ratio, seconds=time.time() - started)


def discover(only: list[str] | None = None) -> list[Path]:
    """visual/ 下待处理的原始网格（跳过已经是简化副本的文件）"""
    meshes = []
    for path in sorted(config.VISUAL_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".stl", ".obj", ".ply"):
            continue
        if config.is_simplified(path):
            continue
        if only and config.clean_stem(path.stem) not in only:
            continue
        meshes.append(path)
    return meshes


def simplify_all(only: list[str] | None = None, max_faces: int = config.MAX_MESH_FACES,
                 force: bool = False) -> dict[str, dict]:
    """批量简化，返回 ``{stem: 报告}``"""
    config.ensure_dirs()
    meshes = discover(only)
    if not meshes:
        print(f"WARNING: {config.VISUAL_DIR} 下没有待处理的网格")
        return {}

    _, usable = _check_backends()
    preferred, quality = usable[0], BACKENDS[usable[0]][1]

    print(f"简化网格：{len(meshes)} 个，面数上限 {max_faces}")
    print(f"  可用简化后端：{', '.join(usable)}；本次优先 {preferred}"
          f"（{BACKENDS[preferred][0]}）")
    if quality < 2:
        print("  WARNING: 没装 fast_simplification / open3d，形状会明显失真；"
              "建议 pip install fast-simplification 后重跑 --force")

    report = {}
    for path in meshes:
        info = simplify_one(path, max_faces=max_faces, force=force)
        report[info["stem"]] = info

    swapped = [s for s, r in report.items() if r["simplified"]]
    backends = sorted({r["backend"] for r in report.values() if r["simplified"]})
    print(f"完成：{len(report)} 个网格，其中 {len(swapped)} 个生成了简化副本"
          f"{'（' + ', '.join(swapped) + '）' if swapped else ''}"
          f"{'，后端 ' + '/'.join(backends) if backends else ''}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将 STL 简化到 MuJoCo 可接受的面数（自动选择可用的简化后端）。")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些网格 stem")
    parser.add_argument("--max-faces", type=int, default=config.MAX_MESH_FACES,
                        help=f"面数上限（默认 {config.MAX_MESH_FACES}）")
    parser.add_argument("--force", action="store_true", help="已有简化副本也重新算")
    args = parser.parse_args()

    simplify_all(only=args.only, max_faces=args.max_faces, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
