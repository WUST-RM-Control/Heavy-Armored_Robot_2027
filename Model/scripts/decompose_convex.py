"""凸包分解：用 CoACD 把每个连杆网格切成若干凸块。

流程里的第 [2] 步，输入是 ``simplify_mesh.py`` 的输出（原件或简化副本）。

输出::

    meshs/colission/LINK1_0.obj, LINK1_1.obj, ...      # 一块一个文件，平铺

* 文件就在网格自身坐标系里，和 STL 同一套坐标，所以 URDF/MJCF 里不需要额外
  的 origin 偏移；
* 每块都保证是凸的，MuJoCo 的 mesh geom 会直接取它的凸包，不会退化；
* 参数按连杆名匹配 ``config.CONVEX_PROFILES``（link / base / finger / peg ...），
  换机器人时在 config 里加规则即可。

用法::

    python scripts/decompose_convex.py                 # 全部
    python scripts/decompose_convex.py --only LINK3 --jobs 2
    python scripts/decompose_convex.py --profile link --threshold 0.05
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import coacd
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def params_for(stem: str, overrides: dict | None = None) -> dict:
    """连杆 → CoACD 参数字典（含 preprocess_mode）"""
    profile = config.profile_for(stem)
    params = dict(config.CONVEX_PROFILES[profile])
    params["preprocess_mode"] = config.CONVEX_PREPROCESS_MODE
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def decompose_one(stem: str, overrides: dict | None = None,
                  force: bool = False) -> list[Path]:
    """分解一个连杆，返回凸包 obj 列表"""
    source = config.visual_mesh(stem)          # 优先用简化后的 STL
    parts = config.collision_parts(stem)
    if parts and not force:
        print(f"  {stem:10s} 已有 {len(parts)} 块凸包，跳过（--force 可重算）")
        return parts

    for stale in config.COLLISION_DIR.glob(f"{stem}_*.obj"):
        stale.unlink()                          # 清掉旧结果，避免序号对不上

    params = params_for(stem, overrides)
    started = time.time()
    mesh = trimesh.load(str(source), force="mesh")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    coacd.set_log_level("warning")      # C++ 那堆 Processing [x%] 太吵，只留警告
    result = coacd.run_coacd(coacd_mesh, **params)

    config.COLLISION_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for index, (vertices, faces) in enumerate(result):
        path = config.COLLISION_DIR / config.convex_name(stem, index)
        trimesh.Trimesh(vertices=vertices, faces=faces).export(str(path))
        written.append(path)

    print(f"  {stem:10s} {source.name:24s} → {len(written):3d} 块 "
          f"(档位 {config.profile_for(stem)}, {time.time() - started:.1f}s)")
    return written


def decompose_all(only: list[str] | None = None, jobs: int = 1,
                  force: bool = False, overrides: dict | None = None) -> dict[str, list[Path]]:
    """批量分解，返回 ``{stem: [obj 路径]}``"""
    config.ensure_dirs()
    stems = discover(only)
    if not stems:
        print(f"WARNING: {config.VISUAL_DIR} 下没有网格可分解")
        return {}

    print(f"凸包分解：{len(stems)} 个网格，档位规则 {config.PART_PROFILE_RULES}")
    result: dict[str, list[Path]] = {}
    if jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {stem: pool.submit(decompose_one, stem, overrides, force) for stem in stems}
            for stem, future in futures.items():
                result[stem] = future.result()
    else:
        for stem in stems:
            result[stem] = decompose_one(stem, overrides=overrides, force=force)

    total = sum(len(v) for v in result.values())
    print(f"完成：{total} 块凸包，输出到 {config.COLLISION_DIR}")
    return result


def discover(only: list[str] | None = None) -> list[str]:
    """要分解的连杆 stem：visual/ 下的原始网格名（去重、按名字排序）"""
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
    parser = argparse.ArgumentParser(description="用 CoACD 做凸包分解。")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些连杆")
    parser.add_argument("--jobs", type=int, default=1, help="并行进程数（COACD 吃内存，默认 1）")
    parser.add_argument("--force", action="store_true", help="已有结果也重算")
    parser.add_argument("--profile", default=None, help="强制用某个档位（覆盖名字匹配）")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖 threshold")
    parser.add_argument("--max-convex-hull", type=int, default=None, help="覆盖 max_convex_hull")
    parser.add_argument("--resolution", type=int, default=None, help="覆盖 resolution")
    args = parser.parse_args()

    if args.profile:
        # 插一条空前缀规则在最前面：所有连杆都命中同一个档位
        config.PART_PROFILE_RULES.insert(0, ("", args.profile))

    overrides = dict(threshold=args.threshold,
                     max_convex_hull=args.max_convex_hull,
                     resolution=args.resolution)
    decompose_all(only=args.only, jobs=args.jobs, force=args.force, overrides=overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
