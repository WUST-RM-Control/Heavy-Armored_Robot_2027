"""球体近似（foam）：调用 **foam 自带的 CLI** 给每个连杆生成球体覆盖。

流程里的第 [3] 步，**取代原来的胶囊（Pinocchio capsule-approximation）近似**
（``decompose_capsule.py`` 代码保留、默认不再调用）。

这个脚本自己不含任何球体生成算法——算法全在 foam 里，这里只做四件事：

1. **调 CLI**：把 ``generate_spheres.py``（单网格，默认）或
   ``generate_sphere_urdf.py``（整份 URDF）当子进程调起来，参数全部走命令行；
2. **选层**：foam 输出的是球树（每层一套球），取 ``--level``（默认最深层）；
3. **剪枝**：丢掉「完全被另一个球包住」的球（无损：并集不变，纯粹少几个 geom）；
4. **复验 + 补半径**：用**原始**网格的表面点检查「网格 ⊆ 球的并集」，
   没盖住的地方按最大穿透量整体放大半径，再复验一遍，保证不漏碰。

foam 出处：https://github.com/CoMMALab/foam
论文：*Foam: A Tool for Spherical Approximation of Robot Geometry*（arXiv:2503.13704）
foam 底层用的是 spheretree（Trinity College Dublin, 2003）的中轴/八叉树等球树算法。

环境是两级的
------------
因为 foam 需要 ``fire`` / ``xmltodict`` / ``vhacdx``，而主环境 ``~/mujoco_env``
里没有，所以：

* **本脚本**只依赖 numpy/trimesh，跑在 ``~/mujoco_env``（和别的步骤一致）；
* **foam CLI** 用 ``FOAM_PYTHON`` 指定的解释器跑，默认自动探测，优先级：
  ``--python`` > ``config.FOAM_PYTHON`` > 当前解释器 > ``<工程>/.pixi/envs/default/bin/python``
  > ``PATH`` 里的 ``python3``。工程根目录的 ``pixi.toml`` 已经声明了
  ``foam = { git = ... }``，所以默认走 pixi 环境。
* foam 的 C++ 二进制（``makeTreeMedial`` 等）不在 pip 包里，需要编译一次：
  ``python scripts/setup_foam.py``（本脚本检测到缺失会直接告诉你这条命令）。

输出::

    meshs/sphere/LINK1_sphere.json    # 球列表 + 位姿/贴合度/复验结果 + 用的命令行
    meshs/sphere/sphere_params.json   # 汇总

用法::

    python scripts/decompose_sphere.py                        # 默认 medial、depth=3
    python scripts/decompose_sphere.py --only LINK4 --depth 2
    python scripts/decompose_sphere.py --method octree --branch 4
    python scripts/decompose_sphere.py --mode urdf --urdf urdf/wust_arm2.urdf
    python scripts/decompose_sphere.py --foam-arg -numCover=8000   # 透传任意参数
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

#: foam CLI 依赖的 C++ 二进制（编译产物，见 setup_foam.py）
BINARIES = (
    "makeTreeMedial", "makeTreeGrid", "makeTreeHubbard", "makeTreeOctree",
    "makeTreeSpawn", "manifold", "manifold_old", "simplify", "simplify_old",
)
#: foam 支持的球树算法
METHODS = ("medial", "grid", "spawn", "octree", "hubbard")

#: 关掉中间提示（run_pipeline 里默认开）
QUIET = False
_PRINT_LOCK = threading.Lock()
_RUNTIME: dict | None = None


# --------------------------------------------------------------------------- #
# 找 foam 运行环境
# --------------------------------------------------------------------------- #
def _restore_cached_binaries(external: Path, quiet: bool = False) -> list[str]:
    """从项目缓存恢复 foam 的 C++ 工具。

    foam 的 pip 包只包含 Python 代码；spheretree 工具是运行时需要的本地
    可执行文件，且 pixi 重建环境时会清掉 site-packages 中的手工文件。
    """
    cache = config.MODEL_DIR / "third_party" / "foam_bin"
    available = [name for name in BINARIES if (cache / name).is_file()]
    if not available:
        return []

    external.mkdir(parents=True, exist_ok=True)
    restored = []
    for name in available:
        target = external / name
        if target.is_file():
            continue
        shutil.copy2(cache / name, target)
        target.chmod(0o755)
        restored.append(name)
    if restored and not quiet:
        print(f"  从项目缓存恢复 foam 二进制：{', '.join(restored)}")
    return restored


def _probe(python: Path) -> dict | None:
    """问这个解释器：foam 包在哪、scripts/ 在哪、external/ 里缺哪些二进制"""
    code = (
        "import pathlib, foam;"
        "p = pathlib.Path(foam.__file__).parent;"
        "print(p); print(p.parent / 'scripts'); print(p / 'external')"
    )
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True,
                                text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if len(lines) < 3:
        return None
    package, scripts, external = (Path(line) for line in lines[-3:])
    if not (scripts / config.FOAM_CLI_MESH).is_file():
        return None
    return dict(python=Path(python), package=package, scripts=scripts, external=external,
                missing=[name for name in BINARIES if not (external / name).is_file()])


def candidate_pythons(explicit: str | None = None) -> list[Path]:
    """按优先级列出候选解释器（去重、保序）"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if config.FOAM_PYTHON:
        candidates.append(Path(config.FOAM_PYTHON).expanduser())
    candidates.append(Path(sys.executable))
    # 工程根目录的 pixi 环境（pixi.toml 里声明了 foam）
    candidates.append(config.MODEL_DIR.parent / ".pixi" / "envs" / "default" / "bin" / "python")
    found = shutil.which("python3")
    if found:
        candidates.append(Path(found))

    unique, seen = [], set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_runtime(explicit: str | None = None, quiet: bool = False) -> dict:
    """找到能跑 foam CLI 的解释器（结果缓存，整个进程只探测一次）"""
    global _RUNTIME
    if _RUNTIME is not None and not explicit:
        return _RUNTIME

    tried = []
    for candidate in candidate_pythons(explicit):
        info = _probe(candidate)
        if info is None:
            tried.append(f"{candidate}（没有 foam 或不可用）")
            continue
        # pip/conda 安装 foam 时不会安装 spheretree 的 C++ 工具；优先从
        # 随项目保存的缓存恢复，避免 pixi 重建环境后再次出现 FileNotFoundError。
        if info["missing"]:
            _restore_cached_binaries(info["external"], quiet=quiet)
            info["missing"] = [
                name for name in BINARIES
                if not (info["external"] / name).is_file()
            ]
        _RUNTIME = info
        if info["missing"] and not quiet:
            print(f"  WARNING: foam 的 C++ 二进制缺 {len(info['missing'])} 个："
                  f"{', '.join(info['missing'])}\n"
                  f"           先跑 `python {config.SCRIPTS_DIR.name}/setup_foam.py` 编译安装，"
                  f"否则 CLI 会失败")
        if not quiet:
            print(f"  foam 运行环境：{info['python']}")
            print(f"    foam 包 {info['package']}")
            print(f"    CLI    {info['scripts'] / config.FOAM_CLI_MESH}")
        return info

    detail = "\n".join(f"    - {item}" for item in tried)
    raise SystemExit(
        "找不到能跑 foam CLI 的 python 解释器。依次试过：\n"
        f"{detail}\n"
        "解决办法（任选一个）：\n"
        "  1. 用工程里的 pixi 环境：pixi install（pixi.toml 里已经声明 foam）\n"
        "  2. 给当前环境装：pip install foam fire xmltodict\n"
        "  3. 直接指定：--python /path/to/python 或改 config.FOAM_PYTHON"
    )


# --------------------------------------------------------------------------- #
# 调 foam CLI
# --------------------------------------------------------------------------- #
def call_cli(info: dict, cli_name: str, args: list[str], timeout: int) -> str:
    """跑一个 foam CLI 脚本，返回标准输出（失败就抛出带尾部的异常）"""
    command = [str(info["python"]), str(info["scripts"] / cli_name)] + [str(a) for a in args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-40:])
        raise RuntimeError(f"foam CLI 失败（退出码 {result.returncode}）："
                           f"{' '.join(command)}\n{tail}")
    return output


def mesh_cli_args(mesh: Path, output: Path, options: dict) -> list[str]:
    """``generate_spheres.py`` 的命令行参数（参数名和 foam 脚本里的形参一一对应）"""
    args = [str(mesh), "--output", str(output),
            "--method", options["method"],
            "--depth", options["depth"],
            "--branch", options["branch"],
            "--manifold_leaves", options["manifold_leaves"],
            "--simplify_ratio", options["simplify_ratio"],
            "--testerLevels", options["tester_levels"],
            "--numCover", options["num_cover"],
            "--minCover", options["min_cover"],
            "--initSpheres", options["init_spheres"],
            "--minSpheres", options["min_spheres"],
            "--erFact", options["er_fact"],
            "--maxOptLevel", options["max_opt_level"],
            "--balExcess", options["bal_excess"],
            "--num_samples", options["num_samples"],
            "--min_samples", options["min_samples"],
            f"--expand={options['expand']}",
            f"--merge={options['merge']}",
            f"--burst={options['burst']}",
            f"--optimise={options['optimise']}",
            f"--verify={options['verify']}"]
    args += list(options["extra"])
    return args


def urdf_cli_args(urdf: Path, output: Path, database: Path, options: dict) -> list[str]:
    """``generate_sphere_urdf.py`` 的命令行参数。

    注意它的形参名和单网格脚本不一样（``simplification_ratio`` 而不是
    ``simplify_ratio``，另有 ``threads`` / ``shrinkage`` / ``volume_heuristic_ratio``）。
    """
    args = [str(urdf), "--output", str(output), "--database", str(database),
            "--method", options["method"],
            "--depth", options["depth"],
            "--branch", options["branch"],
            "--threads", options["threads"],
            "--shrinkage", options["shrinkage"],
            "--volume_heuristic_ratio", options["volume_heuristic_ratio"],
            "--manifold_leaves", options["manifold_leaves"],
            "--simplification_ratio", options["simplify_ratio"],
            "--testerLevels", options["tester_levels"],
            "--numCover", options["num_cover"],
            "--minCover", options["min_cover"],
            "--initSpheres", options["init_spheres"],
            "--minSpheres", options["min_spheres"],
            "--erFact", options["er_fact"],
            "--maxOptLevel", options["max_opt_level"],
            "--balExcess", options["bal_excess"],
            "--num_samples", options["num_samples"],
            "--min_samples", options["min_samples"],
            f"--expand={options['expand']}",
            f"--merge={options['merge']}",
            f"--burst={options['burst']}",
            f"--optimise={options['optimise']}",
            f"--verify={options['verify']}"]
    args += list(options["extra"])
    return args


# --------------------------------------------------------------------------- #
# 结果处理
# --------------------------------------------------------------------------- #
def read_levels(path: Path) -> list[dict]:
    """读 foam 输出的球树 json：``[{mean,best,worst,spheres:[{origin,radius}]}, ...]``"""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):                      # 单层（理论上不会出现）
        data = [data]
    levels = []
    for item in data:
        spheres = item.get("spheres", [])
        levels.append(dict(
            mean=float(item.get("mean", 0.0)),
            best=float(item.get("best", 0.0)),
            worst=float(item.get("worst", 0.0)),
            spheres=[dict(origin=[float(v) for v in s["origin"]], radius=float(s["radius"]))
                     for s in spheres],
        ))
    if not levels:
        raise RuntimeError(f"{path} 里没有任何球")
    return levels


def pick_level(levels: list[dict], level: int) -> tuple[int, dict]:
    """按 ``level``（支持负数，-1 = 最深层）取一层"""
    index = level if level >= 0 else len(levels) + level
    if not 0 <= index < len(levels):
        raise IndexError(f"层号 {level} 超范围：一共 {len(levels)} 层")
    return index, levels[index]


def prune_contained(centers: np.ndarray, radii: np.ndarray, eps: float) -> np.ndarray:
    """丢掉「完全被另一个球包住」的球，返回保留的索引。

    判据：``|c_i - c_j| + r_i <= r_j + eps`` —— 球 i 整个落在球 j 里，
    去掉它并集一个点都不少（无损剪枝）。为了稳，按半径从大到小逐个判。
    """
    order = np.argsort(-radii)
    keep: list[int] = []
    for index in order:
        center, radius = centers[index], radii[index]
        if keep:
            kept = np.asarray(keep)
            distance = np.linalg.norm(centers[kept] - center, axis=1)
            if np.any(distance + radius <= radii[kept] + eps):
                continue
        keep.append(int(index))
    return np.asarray(sorted(keep), dtype=int)


def limit_spheres(centers: np.ndarray, radii: np.ndarray, limit: int) -> np.ndarray:
    """球太多时按体积（r³）从大到小保留 ``limit`` 个（0 = 不限）"""
    if limit <= 0 or len(radii) <= limit:
        return np.arange(len(radii))
    return np.sort(np.argsort(-radii ** 3)[:limit])


def surface_points(mesh_path: Path, limit: int, seed: int = 0) -> np.ndarray:
    """网格表面采样点：三角形顶点 + 面心（超过 limit 就随机抽）"""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    points = np.vstack([np.asarray(mesh.vertices, dtype=float),
                        np.asarray(mesh.triangles, dtype=float).mean(axis=1)])
    if len(points) > limit:
        points = points[np.random.default_rng(seed).choice(len(points), limit, replace=False)]
    return points


def coverage(points: np.ndarray, centers: np.ndarray, radii: np.ndarray,
             neighbours: int = 16) -> np.ndarray:
    """每个表面点到球并集的「有符号距离」：``min_i(|p-c_i| - r_i)``。

    返回数组，``.max() > 0`` 就说明有点没被任何球盖住（漏碰风险），值就是穿透量。
    为了不只查最近的球心（半径不一样时最近球心未必是最优球），查 ``neighbours``
    个最近球心取最小。
    """
    from scipy.spatial import cKDTree

    k = int(min(neighbours, len(radii)))
    distance, index = cKDTree(centers).query(points, k=k)
    if k == 1:
        distance, index = distance[:, None], index[:, None]
    return (distance - radii[index]).min(axis=1)


def compensate(points: np.ndarray, centers: np.ndarray, radii: np.ndarray,
               margin: float, neighbours: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """把「没盖住的表面点」按最近球分摊下去，返回 ``(新半径, 每个球补的量)``。

    做法：每个漏点 ``p`` 找到对它最「负责」的球（``|p-c_i| - r_i`` 最小的那个），
    把这个点需要的补量记在该球头上，最后每个球补自己那份最大值 + ``margin``。

    这样**每个表面点都被它对应的那个球盖住**（严格保证），而且不像「全局最大漏点
    统一放大」那样把所有球都吹胖——补量只落在真正需要的那几个球上。
    """
    from scipy.spatial import cKDTree

    k = int(min(neighbours, len(radii)))
    distance, index = cKDTree(centers).query(points, k=k)
    if k == 1:
        distance, index = distance[:, None], index[:, None]
    signed = distance - radii[index]                      # (点, k)
    best = signed.argmin(axis=1)                          # 每个点最接近的球（在 k 个里）
    gap = signed[np.arange(len(points)), best]            # 该点的漏量（>0 = 没盖住）
    owner = index[np.arange(len(points)), best]

    extra = np.zeros(len(radii))
    np.maximum.at(extra, owner, gap)
    extra = np.maximum(extra, 0.0) + margin
    return radii + extra, extra


# --------------------------------------------------------------------------- #
# 单个连杆
# --------------------------------------------------------------------------- #
def display_command(command: list[str]) -> list[str]:
    """命令行里的绝对路径换成相对工程根的相对路径（json 里不留机器路径）"""
    root = config.MODEL_DIR.parent
    shown = []
    for index, item in enumerate(command):
        text = str(item)
        path = Path(text)
        if path.is_absolute():
            try:
                text = str(path.relative_to(root))
            except ValueError:
                text = path.name if index == 0 else text   # 解释器不在工程里就只留文件名
        shown.append(text)
    return shown


def finalize(stem: str, spheres: list[dict], meta: dict, options: dict,
             command: list[str]) -> dict:
    """剪枝 → 复验 → 补半径 → 写 json；返回这个连杆的参数字典"""
    started = time.time()
    centers = np.asarray([s["origin"] for s in spheres], dtype=float)
    radii = np.asarray([s["radius"] for s in spheres], dtype=float)
    raw_count = len(radii)

    pruned = 0
    if config.SPHERE_PRUNE_CONTAINED and raw_count:
        keep = prune_contained(centers, radii, config.SPHERE_PRUNE_EPS)
        pruned = raw_count - len(keep)
        centers, radii = centers[keep], radii[keep]
    limited = 0
    keep = limit_spheres(centers, radii, config.SPHERE_MAX_SPHERES)
    if len(keep) != len(radii):
        limited = len(radii) - len(keep)
        centers, radii = centers[keep], radii[keep]
    if config.SPHERE_MIN_RADIUS > 0:
        radii = np.maximum(radii, config.SPHERE_MIN_RADIUS)

    # 复验用**原始**网格（foam 内部会把网格 manifold 化 + 简化到 20%，
    # 拿原件复验才知道「球到底有没有盖住真实零件」）
    original = config.visual_path(stem)
    points = surface_points(original, config.SPHERE_VERIFY_SAMPLES)
    penetration_before = float(coverage(points, centers, radii).max()) if len(radii) else 0.0
    if options.get("compensate", True):
        # 漏掉的地方按「最近的球」分摊补半径（保证 网格表面 ⊆ 球并集），再加 margin
        radii, extra = compensate(points, centers, radii, options["margin"])
    else:
        # 不补：只留 margin，漏的地方就是碰撞盲区（json 里的 penetration_after 会报出来）
        extra = np.full(len(radii), float(options["margin"]))
        radii = radii + extra
    penetration_after = float(coverage(points, centers, radii).max()) if len(radii) else 0.0

    hull_volume = meta.get("hull_volume", float("nan"))
    sphere_volume = float((4.0 / 3.0 * np.pi * radii ** 3).sum())

    info = dict(
        stem=stem,
        source=meta.get("source"),
        cli=meta.get("cli"),
        command=display_command(command),
        method=options["method"],
        depth=options["depth"],
        branch=options["branch"],
        level=meta.get("level"),
        levels=meta.get("levels"),
        error=dict(mean=meta.get("mean"), best=meta.get("best"), worst=meta.get("worst")),
        # 球
        nspheres=int(len(radii)),
        nspheres_raw=int(raw_count),
        pruned=int(pruned),
        limited=int(limited),
        margin=float(options["margin"]),
        radius_added=float(extra.max()) if len(radii) else 0.0,
        radius_added_mean=float(extra.mean()) if len(radii) else 0.0,
        radius_min=float(radii.min()) if len(radii) else 0.0,
        radius_max=float(radii.max()) if len(radii) else 0.0,
        radius_sum=float(radii.sum()) if len(radii) else 0.0,
        sphere_volume=sphere_volume,
        hull_volume=float(hull_volume),
        fill_ratio=float(sphere_volume / hull_volume) if hull_volume else float("nan"),
        # 复验（对原始网格）
        verify=dict(points=int(len(points)),
                    penetration_before=penetration_before,
                    penetration_after=penetration_after,
                    clearance=-penetration_after),
        spheres=[dict(origin=[float(v) for v in center], radius=float(radius))
                 for center, radius in zip(centers, radii)],
        seconds=float(meta.get("seconds", 0.0)),
        process_seconds=time.time() - started,
    )

    if config.SPHERE_WRITE_MESH and len(radii):
        info["file"] = write_union_mesh(stem, centers, radii).name

    target = config.sphere_path(stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def write_union_mesh(stem: str, centers: np.ndarray, radii: np.ndarray):
    """把球并集导成 obj（低面数 icosphere 拼起来，只为看图/自查）"""
    import trimesh

    parts = []
    subdivisions = config.SPHERE_MESH_SUBDIVISIONS
    for center, radius in zip(centers, radii):
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=float(radius))
        sphere.apply_translation(center)
        parts.append(sphere)
    path = config.SPHERE_DIR / f"{stem}_sphere.obj"
    trimesh.util.concatenate(parts).export(path)
    return path


def hull_volume_of(stem: str) -> float:
    """连杆凸包体积——用来算「球体占了凸包的多少」，保守度的粗指标"""
    import trimesh

    mesh = trimesh.load(str(config.visual_path(stem)), force="mesh", process=False)
    try:
        return float(abs(trimesh.convex.convex_hull(mesh.vertices).volume))
    except Exception:
        return float("nan")


def report_line(info: dict) -> str:
    verify = info["verify"]
    return (f"  {info['stem']:10s} {str(info['source']):24s} "
            f"球 {info['nspheres']:4d}（原始 {info['nspheres_raw']:4d}"
            f"{'，剪掉 ' + str(info['pruned']) if info['pruned'] else ''}） "
            f"r {info['radius_min'] * 1000:6.1f}~{info['radius_max'] * 1000:6.1f}mm "
            f"净空 {verify['clearance'] * 1000:+.3f}mm "
            f"（补半径 {info['radius_added'] * 1000:+.3f}mm，"
            f"{info['seconds']:.1f}s）")


# --------------------------------------------------------------------------- #
# 两种模式
# --------------------------------------------------------------------------- #
def discover(only: list[str] | None = None) -> list[str]:
    """要近似的连杆 stem（和凸分解一致：去重排序、跳过简化副本）"""
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


def sphere_one(stem: str, options: dict, force: bool = False) -> dict:
    """单个连杆：调一次 CLI → 处理 → 写 json"""
    target = config.sphere_path(stem)
    if target.is_file() and not force:
        if not QUIET:
            print(f"  {stem:10s} 已有 {target.name}，跳过（--force 可重算）")
        return json.loads(target.read_text())

    runtime = resolve_runtime(options.get("python"), quiet=True)
    source = config.visual_mesh(stem)
    raw_json = config.SPHERE_DIR / f".{stem}_foam_raw.json"
    started = time.time()
    command = mesh_cli_args(source, raw_json, options)
    with _PRINT_LOCK:
        print(f"  {stem:10s} foam CLI ← {source.name}（method={options['method']} "
              f"depth={options['depth']} branch={options['branch']}）…", flush=True)
    output = call_cli(runtime, config.FOAM_CLI_MESH, command, options["timeout"])
    levels = read_levels(raw_json)
    raw_json.unlink(missing_ok=True)
    index, level = pick_level(levels, options["level"])
    if not level["spheres"]:
        raise RuntimeError(f"{stem} 的第 {index} 层一个球都没有")

    info = finalize(stem, level["spheres"], dict(
        source=source.name, cli=config.FOAM_CLI_MESH, level=index,
        levels=[len(item["spheres"]) for item in levels],
        mean=level["mean"], best=level["best"], worst=level["worst"],
        hull_volume=hull_volume_of(stem), seconds=time.time() - started,
    ), options, command)
    with _PRINT_LOCK:
        print(report_line(info), flush=True)
        if not QUIET and "ERROR" in output.upper():
            print(f"    NOTE: CLI 输出里有 ERROR 字样，自己确认一下："
                  f"{output.strip().splitlines()[-1] if output.strip() else ''}")
    return info


def spheres_from_urdf(urdf_path: Path) -> dict[str, list[dict]]:
    """从 foam 生成的「全球体 URDF」里把球按连杆抠出来 → ``{stem: [球]}``"""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    result: dict[str, list[dict]] = {}
    for link in root.findall("link"):
        name = link.get("name", "")
        stem = name
        for mesh in link.iter("mesh"):                   # 用视觉网格名反推 stem
            filename = mesh.get("filename")
            if filename:
                stem = config.clean_stem(config.stem_of(filename))
                break
        spheres = []
        for collision in link.findall("collision"):
            sphere = collision.find("./geometry/sphere")
            if sphere is None:
                continue
            origin = collision.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None
                                      else [0.0, 0.0, 0.0])]
            spheres.append(dict(origin=xyz, radius=float(sphere.get("radius"))))
        if spheres:
            result[stem] = spheres
    return result


def sphere_all_urdf(options: dict, urdf_path: Path, only: list[str] | None = None) -> dict[str, dict]:
    """``--mode urdf``：一次 CLI 调用覆盖整份 URDF，再把结果按连杆拆开。

    这条路径**总是重算**（CLI 是整份 URDF 一次跑完的，没有按连杆跳过的余地）。

    ⚠ 它近似的是 **URDF 里现有的 <collision> 几何**：本工程的
    ``urdf/<robot>.urdf`` 用的是 box/cylinder/sphere 图元，所以这条路径得到的是
    「图元模型的球体近似」，不是「视觉网格的球体近似」。想要后者请用默认的
    ``--mode mesh``，或者把 ``--urdf`` 指到 ``_convex.urdf``（网格碰撞）上。
    """
    runtime = resolve_runtime(options.get("python"), quiet=True)
    out_urdf = config.SPHERE_DIR / ".foam_spherized.urdf"
    database = config.SPHERE_DIR / ".foam_sphere_database.json"
    command = urdf_cli_args(urdf_path, out_urdf, database, options)
    print(f"  foam CLI ← {urdf_path.name}（整份 URDF，method={options['method']} "
          f"depth={options['depth']} branch={options['branch']}）…", flush=True)
    output = call_cli(runtime, config.FOAM_CLI_URDF, command, options["timeout"])
    if not QUIET and output.strip():
        print("    " + output.strip().splitlines()[-1])

    per_link = spheres_from_urdf(out_urdf)
    out_urdf.unlink(missing_ok=True)
    database.unlink(missing_ok=True)

    report = {}
    for stem in discover(only):
        spheres = per_link.get(stem)
        if not spheres:
            print(f"  WARNING: {stem} 在 {urdf_path.name} 里没有球体结果，跳过")
            continue
        info = finalize(stem, spheres, dict(
            source=f"{urdf_path.name}（{config.FOAM_CLI_URDF}）", cli=config.FOAM_CLI_URDF,
            level=None, levels=[len(spheres)],
            mean=None, best=None, worst=None,
            hull_volume=hull_volume_of(stem), seconds=0.0,
        ), options, command)
        report[stem] = info
        print(report_line(info))
    return report


def sphere_all(only: list[str] | None = None, force: bool = False, options: dict | None = None,
               mode: str = "mesh", urdf: Path | None = None) -> dict[str, dict]:
    """批量生成，返回 ``{stem: 参数}`` 并写出 sphere_params.json"""
    global QUIET
    options = dict(options or {})
    QUIET = bool(options.get("quiet", QUIET))
    options.setdefault("quiet", QUIET)
    config.ensure_dirs()
    failures: dict[str, Exception] = {}

    print(f"球体近似（foam / {config.FOAM_CLI_MESH if mode == 'mesh' else config.FOAM_CLI_URDF}）："
          f"算法 {options['method']}，depth={options['depth']}，branch={options['branch']}，"
          f"并行 {options['jobs']}")
    if mode == "urdf":
        report = sphere_all_urdf(options, urdf or config.urdf_path(), only=only)
    else:
        stems = discover(only)
        if not stems:
            print(f"WARNING: {config.VISUAL_DIR} 下没有网格可做球体近似")
            return {}
        if only:
            missing = [s for s in only if s not in stems]
            if missing:
                print(f"  WARNING: 找不到这些连杆：{', '.join(missing)}")
        report: dict[str, dict] = {}
        jobs = max(1, int(options["jobs"]))
        started = time.time()
        if jobs == 1:
            for stem in stems:
                try:
                    report[stem] = sphere_one(stem, options, force=force)
                except Exception as exc:                # 一个连杆失败不拖垮其余连杆
                    failures[stem] = exc
                    print(f"  ✗ {stem} 失败：{exc}")
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(sphere_one, stem, options, force): stem for stem in stems}
                for future in as_completed(futures):
                    stem = futures[future]
                    try:
                        report[stem] = future.result()
                    except Exception as exc:
                        failures[stem] = exc
                        print(f"  ✗ {stem} 失败：{exc}")
        report = {stem: report[stem] for stem in stems if stem in report}
        print(f"  合计 {time.time() - started:.1f}s")

    if config.SPHERE_WRITE_PARAMS and report:
        params_file = config.sphere_params_path()
        merged = report
        if only and params_file.is_file():
            # 只跑了一部分连杆：别把其它连杆的汇总覆盖掉，合并进去
            existing = json.loads(params_file.read_text())
            existing.update(report)
            merged = {stem: existing[stem] for stem in sorted(existing)}
        params_file.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"参数已写入 {params_file.relative_to(config.MODEL_DIR)}（{len(merged)} 个连杆）")

    total = sum(item["nspheres"] for item in report.values())
    print(f"完成：{len(report)} 个连杆、共 {total} 个球，输出到 "
          f"{config.SPHERE_DIR.relative_to(config.MODEL_DIR)}/")
    if failures:
        raise SystemExit(f"{len(failures)} 个连杆没跑出来：{', '.join(failures)}"
                         f"（换个 --method/--depth，或者把它们加进 config.SPHERE_SKIP_LINKS）")
    return report


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 foam 的 CLI 给连杆生成球体近似（取代胶囊近似）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些连杆")
    parser.add_argument("--force", action="store_true", help="已有结果也重算")
    parser.add_argument("--mode", choices=("mesh", "urdf"), default="mesh",
                        help="mesh = 逐网格调 generate_spheres.py；"
                             "urdf = 调 generate_sphere_urdf.py 处理整份 URDF")
    parser.add_argument("--urdf", default=None, help="--mode urdf 时要近似的 URDF")
    # --- foam / spheretree 参数（默认值全部来自 config） ---
    parser.add_argument("--method", choices=METHODS, default=config.FOAM_METHOD,
                        help="球树算法")
    parser.add_argument("--depth", type=int, default=config.FOAM_DEPTH,
                        help="球树层数（层数 = depth+1）")
    parser.add_argument("--branch", type=int, default=config.FOAM_BRANCH,
                        help="分支因子（最深层最多 branch**depth 个球）")
    parser.add_argument("--level", type=int, default=config.SPHERE_LEVEL,
                        help="取第几层（负数从末尾数，-1 = 最深层）")
    parser.add_argument("--margin", type=float, default=config.SPHERE_MARGIN,
                        help="半径余量（米）")
    parser.add_argument("--compensate", action=argparse.BooleanOptionalAction,
                        default=config.SPHERE_COMPENSATE,
                        help="复验后按最近的球分摊补半径，保证网格被完全盖住")
    parser.add_argument("--manifold-leaves", dest="manifold_leaves", type=int,
                        default=config.FOAM_MANIFOLD_LEAVES, help="manifold 的目标叶子数")
    parser.add_argument("--simplify-ratio", dest="simplify_ratio", type=float,
                        default=config.FOAM_SIMPLIFY_RATIO, help="网格预处理简化比例")
    parser.add_argument("--tester-levels", dest="tester_levels", type=int,
                        default=config.FOAM_TESTER_LEVELS)
    parser.add_argument("--num-cover", dest="num_cover", type=int, default=config.FOAM_NUM_COVER)
    parser.add_argument("--min-cover", dest="min_cover", type=int, default=config.FOAM_MIN_COVER)
    parser.add_argument("--init-spheres", dest="init_spheres", type=int,
                        default=config.FOAM_INIT_SPHERES, help="medial：初始球数")
    parser.add_argument("--min-spheres", dest="min_spheres", type=int,
                        default=config.FOAM_MIN_SPHERES, help="medial：子区域最少球数")
    parser.add_argument("--er-fact", dest="er_fact", type=int, default=config.FOAM_ER_FACT)
    parser.add_argument("--max-opt-level", dest="max_opt_level", type=int,
                        default=config.FOAM_MAX_OPT_LEVEL)
    parser.add_argument("--bal-excess", dest="bal_excess", type=float, default=config.FOAM_BAL_EXCESS)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=config.FOAM_NUM_SAMPLES,
                        help="hubbard：采样点数")
    parser.add_argument("--min-samples", dest="min_samples", type=int, default=config.FOAM_MIN_SAMPLES)
    parser.add_argument("--shrinkage", type=float, default=config.FOAM_SHRINKAGE,
                        help="--mode urdf：碰撞体缩放")
    parser.add_argument("--threads", type=int, default=config.FOAM_THREADS,
                        help="--mode urdf：CLI 内部线程数")
    parser.add_argument("--volume-heuristic-ratio", dest="volume_heuristic_ratio", type=float,
                        default=0.7, help="--mode urdf：按体积自动定 branch 的比例")
    for flag, default, help_text in (
        ("expand", config.FOAM_EXPAND, "medial：EXPAND 优化"),
        ("merge", config.FOAM_MERGE, "medial：MERGE 优化"),
        ("burst", config.FOAM_BURST, "medial：BURST 优化"),
        ("optimise", config.FOAM_OPTIMISE, "非 medial：simplex 再优化"),
        ("verify", config.FOAM_VERIFY, "让 spheretree 再校验一遍模型（慢，且 depth>=3 容易失败）"),
    ):
        parser.add_argument(f"--{flag}", dest=flag, action=argparse.BooleanOptionalAction,
                            default=default, help=help_text)
    # --- 环境 / 执行 ---
    parser.add_argument("--jobs", type=int, default=config.FOAM_JOBS,
                        help="并行跑几个连杆的 CLI")
    parser.add_argument("--timeout", type=int, default=config.FOAM_TIMEOUT,
                        help="单次 CLI 超时（秒）")
    parser.add_argument("--python", default=None,
                        help="跑 foam CLI 的 python（默认自动探测，见模块文档）")
    parser.add_argument("--foam-arg", dest="extra", action="append", default=[],
                        help="透传给 foam CLI 的额外参数，可重复（如 --foam-arg -numCover=8000）")
    parser.add_argument("--quiet", action="store_true", help="少打印")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    options = {key: value for key, value in vars(args).items()
               if key not in ("only", "force", "mode", "urdf")}
    sphere_all(only=args.only, force=args.force, options=options, mode=args.mode,
               urdf=Path(args.urdf).expanduser().resolve() if args.urdf else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
