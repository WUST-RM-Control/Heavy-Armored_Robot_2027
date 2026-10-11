"""编译 foam 的 C++ 二进制并装进 foam 包（跑通 ``decompose_sphere.py`` 的前置条件）。

为什么需要这个脚本
------------------
foam 的 pip 包（``pip install git+https://github.com/CoMMALab/foam``）**只带 python
部分**，球树算法在 C++ 里（foam 自带的 ``external/spheretree``，Trinity College
Dublin 2003），必须本地编译一次。foam 期望这些二进制放在
``<site-packages>/foam/external/`` 下，文件名不带后缀：

    makeTreeMedial  makeTreeGrid  makeTreeHubbard  makeTreeOctree  makeTreeSpawn
    manifold  manifold_old  simplify  simplify_old

本脚本按「**缓存 → 编译**」两级来补：

1. 先看 ``--cache-dir``（默认 ``Model/third_party/foam_bin``）里有没有现成的
   二进制——有就直接拷进 foam 包（几百毫秒，换机器/重装 pixi 环境后特别有用）；
2. 缓存也没有就编译：源码默认放在 ``<工程根>/.foam-src/foam``，没有就
   ``git clone --recurse-submodules``（需要联网，约 400MB），然后 cmake + ninja
   编译（几分钟），编完把二进制**同时**装进 foam 包和缓存目录。

``pixi install`` / 重装环境后 ``foam/external/`` 会被还原成空的，这时候只要跑
``python scripts/setup_foam.py --install-only`` 就能从缓存秒装回来。

用法::

    python scripts/setup_foam.py                 # 缺什么补什么
    python scripts/setup_foam.py --check         # 只看状态（不全就退出码 1）
    python scripts/setup_foam.py --install-only  # 只从缓存装，不编译
    python scripts/setup_foam.py --force         # 重新编译
    python scripts/setup_foam.py --jobs 8 --python /path/to/python
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from decompose_sphere import BINARIES, resolve_runtime  # noqa: E402

#: foam 源码仓库
REPO = "https://github.com/CoMMALab/foam.git"
#: 源码默认落地目录（放工程根，不进 Model/，免得 Model 变成几百 MB）
DEFAULT_SRC = config.MODEL_DIR.parent / ".foam-src" / "foam"
#: 二进制缓存目录（跟着 Model 走，方便整体搬家）
DEFAULT_CACHE = config.MODEL_DIR / "third_party" / "foam_bin"


def missing_binaries(directory: Path) -> list[str]:
    return [name for name in BINARIES if not (directory / name).is_file()]


def run(command: list[str], cwd: Path | None = None, quiet: bool = False) -> int:
    if not quiet:
        print(f"    $ {' '.join(str(c) for c in command)}")
    result = subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None)
    return result.returncode


def ensure_source(src: Path, quiet: bool = False) -> bool:
    """确保源码 + 子模块都在（子模块里有 eigen/libigl/manifold/simplify）"""
    if not (src / "CMakeLists.txt").is_file():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"  克隆 foam 源码 → {src}（递归拉子模块，约 400MB，需要联网）")
        code = run(["git", "clone", "--depth", "1", "--recurse-submodules",
                    "--shallow-submodules", REPO, str(src)], quiet=quiet)
        if code != 0:
            print(f"  ✗ 克隆失败（退出码 {code}）")
            return False
    else:
        print(f"  复用已有源码 {src}")
        # 子模块目录是空的就补一下
        if not any((src / "external" / name).iterdir()
                   for name in ("manifold", "simplify", "manifold_old")
                   if (src / "external" / name).is_dir()):
            print("  补拉子模块 …")
            if run(["git", "submodule", "update", "--init", "--recursive",
                    "--depth", "1"], cwd=src, quiet=quiet) != 0:
                print("  ✗ 子模块拉取失败")
                return False
    return True


def build(src: Path, build_dir: Path, jobs: int, quiet: bool = False) -> bool:
    """cmake 配置 + 编译（把二进制编到 ``<build>/external/``）"""
    print(f"  cmake 配置 → {build_dir}")
    if run(["cmake", "-B", str(build_dir), "-GNinja", "-DCMAKE_BUILD_TYPE=Release",
            str(src)], quiet=quiet) != 0:
        print("  ✗ cmake 配置失败（要 cmake >= 3.1 和 ninja）")
        return False
    print(f"  编译（-j {jobs}）…")
    if run(["cmake", "--build", str(build_dir), "-j", str(jobs)], quiet=quiet) != 0:
        print("  ✗ 编译失败")
        return False
    return True


def binary_candidates(build_dir: Path, src: Path, name: str) -> list[Path]:
    """二进制可能在三个地方（cmake 的收尾步骤会把它们汇总到 src/foam/external）"""
    return [
        build_dir / "external" / name,              # manifold / simplify / ...
        build_dir / "external" / "spheretree" / name,   # makeTree*（spheretree 子工程）
        src / "foam" / "external" / name,           # cmake 收尾拷贝的汇总目录
    ]


def install(build_dir: Path, src: Path, target: Path, cache: Path | None = None) -> list[str]:
    """把编出来的二进制装进 foam 包（顺便存一份到缓存）"""
    installed = []
    target.mkdir(parents=True, exist_ok=True)
    for name in BINARIES:
        candidate = next((p for p in binary_candidates(build_dir, src, name) if p.is_file()), None)
        if candidate is None:
            print(f"  WARNING: 没编出 {name}")
            continue
        shutil.copy2(candidate, target / name)
        (target / name).chmod(0o755)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, cache / name)
            (cache / name).chmod(0o755)
        installed.append(name)
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description="编译/安装 foam 的 C++ 二进制。")
    parser.add_argument("--python", default=None, help="foam 所在的解释器（默认自动探测）")
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="foam 源码目录")
    parser.add_argument("--build-dir", default=None, help="编译目录（默认 <src>/build）")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="二进制缓存目录")
    parser.add_argument("--jobs", type=int, default=8, help="编译并行数")
    parser.add_argument("--install-only", action="store_true", help="只从缓存安装，不编译")
    parser.add_argument("--no-cache", action="store_true", help="不读也不写缓存")
    parser.add_argument("--force", action="store_true", help="已装好也重新编译")
    parser.add_argument("--check", action="store_true", help="只检查状态，不安装")
    parser.add_argument("--quiet", action="store_true", help="少打印 cmake 命令")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    build_dir = Path(args.build_dir).expanduser() if args.build_dir else src / "build"
    cache = None if args.no_cache else Path(args.cache_dir).expanduser()

    print("=" * 78)
    print("foam 运行环境自检")
    print("=" * 78)
    runtime = resolve_runtime(args.python, quiet=False)
    target = runtime["external"]
    cached = [] if cache is None else [n for n in BINARIES if (cache / n).is_file()]
    missing = missing_binaries(target)
    print(f"  foam 包      {runtime['package']}")
    print(f"  二进制目录   {target}")
    print(f"  缓存目录     {cache if cache else '（未启用）'}"
          f"{f'，已有 {len(cached)}/{len(BINARIES)}' if cache else ''}")
    print(f"  缺失         {', '.join(missing) if missing else '无（都齐了）'}")

    if args.check:
        return 1 if missing else 0
    if not missing and not args.force:
        print("\n  已经装好了，不用动（--force 可以重编）。")
        return 0

    # ---- 从缓存装 ----
    if cache is not None and len(cached) == len(BINARIES) and not args.force:
        print(f"\n从缓存安装 {len(cached)} 个二进制 → {target}")
        for name in BINARIES:
            shutil.copy2(cache / name, target / name)
            (target / name).chmod(0o755)
        print(f"  完成：{len(BINARIES)} 个二进制就位。")
        return 0
    if args.install_only:
        print(f"\n  ✗ --install-only 但缓存里不全（{len(cached)}/{len(BINARIES)}），"
              f"去掉这个参数就地编译，或者先跑一次完整安装。")
        return 1

    # ---- 编译 ----
    print(f"\n开始编译（源码 {src}）")
    if not ensure_source(src, quiet=args.quiet):
        return 1
    if not build(src, build_dir, args.jobs, quiet=args.quiet):
        return 1
    installed = install(build_dir, src, target, cache=cache)
    print(f"\n  装好 {len(installed)}/{len(BINARIES)} 个二进制 → {target}")
    if cache is not None:
        print(f"  同时缓存到 {cache}（下次 --install-only 秒装）")
    remaining = missing_binaries(target)
    if remaining:
        print(f"  ✗ 还缺：{', '.join(remaining)}")
        return 1
    print("\n环境齐了，可以跑： python scripts/decompose_sphere.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
