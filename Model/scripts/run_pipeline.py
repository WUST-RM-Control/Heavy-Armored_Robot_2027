"""一键流水线：原始 URDF + STL → 3 个 URDF + 4 个 MJCF。

    [0] import   把源模型（URDF + 它引用的 STL）拷进 Model/，路径改写成相对路径
    [1] simplify STL 简化到 10 万面以内              → meshs/simplified/
    [2] convex   CoACD 凸包分解                      → meshs/colission/
    [3] sphere   foam 球体近似（调用 foam 的 CLI）    → meshs/sphere/
    [4] urdf     生成 _convex.urdf / _sphere.urdf    → urdf/
    [5] mjcf     转换 + 视觉网格 + 接触排除 + 执行器  → mjcf/（4 套）
    [6] verify   逐个加载、跑一小段仿真做体检

**第 [3] 步现在是 foam 球体近似**：``decompose_sphere.py`` 直接调 foam
（https://github.com/CoMMALab/foam ）自带的 ``generate_spheres.py`` /
``generate_sphere_urdf.py``，用 spheretree 的球树算法给出「一堆球」的碰撞体。
它取代了原来的胶囊近似：胶囊（Pinocchio capsule-approximation）代码保留在
``decompose_capsule.py``、可以单独运行，但默认不再调用——一个连杆一个胶囊对
这种弯折连杆太胖（实测零位下 11 对幻影接触、最深穿透 202mm），球体覆盖可以
按需加密，贴合度和凸包是一个量级的。更早的 OBB 同理（``decompose_obb.py``）。

**每一步都是一个独立子进程**（``subprocess`` 调对应脚本），不是本进程 import：

* ``coacd`` 和 ``open3d`` 在同一个进程里 import 会段错误（实测 exit 139，
  顺序无关、偶尔还看运气），分进程是唯一稳妥的隔离方式；
* foam 的 CLI 还需要另一套 python 环境（``fire``/``xmltodict``/``vhacdx`` +
  编译好的 C++ 二进制），``decompose_sphere.py`` 内部会用它自己的解释器去调；
* 每步的日志、退出码分得清清楚楚，崩在哪一步一目了然；
* COACD 很吃内存，子进程结束即归还，跑大机器人时不容易 OOM。

用法::

    # 用 Model/urdf 里已有的原始 URDF
    python scripts/run_pipeline.py

    # 第一次导入别的机器人（URDF + 网格会被拷进 Model/，之后就能独立跑）
    python scripts/run_pipeline.py --urdf /path/to/wust_arm2.urdf

    # 只重做某几步 / 强制重算 / 并行
    python scripts/run_pipeline.py --steps sphere urdf mjcf
    python scripts/run_pipeline.py --force --jobs 2

    # 想把老变体加回来（对照用）：--steps capsule obb
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

#: 默认跑的步骤（``capsule``/``obb`` 不在里面：代码保留，要跑就 --steps 显式点名）
STEPS = ("import", "simplify", "convex", "sphere", "capsule", "obb", "urdf", "mjcf", "verify")
DEFAULT_STEPS = ("import", "simplify", "convex", "sphere", "urdf", "mjcf", "verify")


# --------------------------------------------------------------------------- #
# 子进程调度
# --------------------------------------------------------------------------- #
def run_step(title: str, script: str, extra: list[str] | None = None,
             keep_going: bool = False) -> bool:
    """跑一个脚本子进程，实时把输出转出来"""
    command = [sys.executable, "-u", str(config.SCRIPTS_DIR / script)] + list(extra or [])
    print("\n" + "=" * 78)
    print(f"{title}：{' '.join(Path(c).name if i == 0 else c for i, c in enumerate(command))}")
    print("=" * 78, flush=True)
    result = subprocess.run(command, cwd=str(config.MODEL_DIR))
    if result.returncode != 0:
        print(f"!! {script} 退出码 {result.returncode}")
        if not keep_going:
            raise SystemExit(f"{script} 失败（退出码 {result.returncode}），流水线中止")
        return False
    return True


# --------------------------------------------------------------------------- #
# [0] 导入源模型
# --------------------------------------------------------------------------- #
def _resolve_source_mesh(filename: str, urdf_dir: Path, search_root: Path) -> Path | None:
    """在源工程里找网格：先按 URDF 里的相对路径，再按文件名递归找"""
    name = filename.replace("package://", "").replace("file://", "")
    candidates = [urdf_dir / filename, urdf_dir / name, search_root / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(search_root.rglob(Path(name).name))
    return matches[0] if matches else None


def import_source_model(source_urdf: Path) -> Path:
    """把源 URDF + 网格拷进 Model/，网格路径改写成 ``../meshs/visual/<文件>``。

    拷贝是**平铺**的：只保留文件名，不保留源工程的目录层级——这样 Model 目录
    自洽，换台机器直接能用。原始 URDF 里除 mesh 路径外的内容一个字不改。
    """
    config.ensure_dirs()
    source_urdf = source_urdf.resolve()
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(source_urdf, parser=parser)
    root = tree.getroot()
    robot = root.get("name") or source_urdf.stem

    copies, missing = 0, []
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        resolved = _resolve_source_mesh(filename, source_urdf.parent, source_urdf.parent)
        if resolved is None:
            missing.append(filename)
            continue
        target = config.VISUAL_DIR / resolved.name
        if not target.is_file() or target.stat().st_size != resolved.stat().st_size:
            shutil.copy2(resolved, target)
            copies += 1
        mesh.set("filename", f"../meshs/visual/{resolved.name}")

    target_urdf = config.urdf_path(robot)
    ET.indent(tree, space="  ")
    target_urdf.write_bytes(ET.tostring(root, encoding="utf-8"))
    print(f"导入 {source_urdf} → {target_urdf.relative_to(config.MODEL_DIR)}"
          f"（网格拷贝 {copies} 个，网格目录 {config.VISUAL_DIR.relative_to(config.MODEL_DIR)}）")
    for item in missing:
        print(f"  WARNING: 找不到网格 {item}")
    return target_urdf


# --------------------------------------------------------------------------- #
# [6] 体检
# --------------------------------------------------------------------------- #
def verify(robot: str = config.DEFAULT_ROBOT, steps: int = 400) -> bool:
    """把生成的 3 个 URDF + 4 个 MJCF 都加载一遍，并跑一小段仿真。

    * URDF：MuJoCo 能不能读、有多少 geom、质量对不对得上；
    * MJCF：关节/几何/执行器数量、跑 ``steps`` 步看有没有发散，
      PD 顺便报一下跟踪误差（给 0.3 rad 阶跃），MOTOR 报一下力矩响应；
    * 零位接触：各套碰撞体在 qpos=0 时各有多少接触、最深穿透多少，
      这是判断「碰撞体胖不胖」最直接的指标（接触对也一并列出来）。
    """
    import mujoco
    import numpy as np

    print("\n" + "=" * 78)
    print("体检")
    print("=" * 78)
    ok = True

    mass = {}
    variants = [""] + list(dict.fromkeys([*config.DEFAULT_VARIANTS, *config.LEGACY_VARIANTS]))
    for variant in variants:
        path = config.urdf_path(robot, variant)
        if not path.is_file():
            if not variant or variant in config.DEFAULT_VARIANTS:
                print(f"  ✗ 缺少 {path.name}")
                ok = False
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
            mass[variant] = float(model.body_mass.sum())
            print(f"  ✓ {path.name:32s} nbody={model.nbody} njnt={model.njnt} "
                  f"ngeom={model.ngeom} nmesh={model.nmesh} "
                  f"质量={mass[variant]:.4f}kg")
        except Exception as exc:
            print(f"  ✗ {path.name}: {exc}")
            ok = False

    if len(set(mass.values())) > 1:
        print(f"  ✗ 各变体总质量不一致：{mass}")
        ok = False
    elif mass:
        print(f"  ✓ 所有变体总质量一致（{next(iter(mass.values())):.4f} kg，"
              f"URDF 自带 <inertial>，geom 不参与质量）")

    for collision, actuator in config.MJCF_VARIANTS:
        path = config.mjcf_path(collision, actuator, robot)
        if not path.is_file():
            print(f"  ✗ 缺少 {path.name}")
            ok = False
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            if actuator == "pd":
                # PD：ctrl 就是目标角，给 JOINT1 一个 0.3 rad 阶跃看跟踪
                for _ in range(steps):
                    if model.nu:
                        data.ctrl[:] = 0.3
                    mujoco.mj_step(model, data)
                extra = f"JOINT1 跟到 {float(data.qpos[0]):.4f} rad（目标 0.3）"
            else:
                # MOTOR：ctrl 就是力矩，给 JOINT1 施加 20% 额定力矩看响应
                effort = float(model.actuator_ctrlrange[0][1]) if model.nu else 0.0
                torque = 0.2 * effort
                for _ in range(steps):
                    if model.nu:
                        data.ctrl[:] = 0.0
                        data.ctrl[0] = torque
                    mujoco.mj_step(model, data)
                extra = (f"JOINT1 施加 {torque:.2f} N·m → 转到 {float(data.qpos[0]):.4f} rad，"
                         f"实际力矩 {float(data.actuator_force[0]):.3f} N·m")
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            flag = "✓" if finite else "✗"
            print(f"  {flag} {path.name:32s} ngeom={model.ngeom:3d} nmesh={model.nmesh:3d} "
                  f"nu={model.nu} ncon={data.ncon} {extra}")
            ok &= finite
        except Exception as exc:
            print(f"  ✗ {path.name}: {exc}")
            ok = False

    # 零位接触：碰撞体胖不胖，看这个最直接
    print("\n  零位（qpos=0）接触检查：")
    for collision in ("convex", "sphere", "capsule", "obb"):
        path = config.mjcf_path(collision, "pd", robot)
        if not path.is_file():
            continue
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        deepest, pairs = 0.0, []
        for index in range(data.ncon):
            contact = data.contact[index]
            # MuJoCo 的 dist < 0 就是穿透深度，取最小的那个
            deepest = min(deepest, float(contact.dist))
            names = []
            for geom in contact.geom:
                body = int(model.geom_bodyid[geom])
                names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "world")
            pairs.append(f"{names[0]}↔{names[1]}")
        summary = "、".join(sorted(set(pairs))) if pairs else "无接触"
        print(f"    {collision:8s} ncon={data.ncon:3d} 最深穿透 "
              f"{-deepest * 1000:7.2f}mm  {summary}")

    print("=" * 78)
    print("体检" + ("通过" if ok else "发现问题"))
    return ok


# --------------------------------------------------------------------------- #
# 产出清单
# --------------------------------------------------------------------------- #
def inventory(robot: str) -> None:
    print("\n产出清单：")
    for directory in (config.URDF_DIR, config.MJCF_DIR, config.COLLISION_DIR,
                      config.SPHERE_DIR, config.CAPSULE_DIR, config.OBB_DIR):
        if not directory.is_dir():
            continue
        files = sorted(p for p in directory.glob("*") if p.is_file())
        total = sum(p.stat().st_size for p in files)
        print(f"  {directory.relative_to(config.MODEL_DIR)}/  {len(files):3d} 个文件 "
              f"({total / 1e3:.1f} kB)")
        for path in files[:6]:
            print(f"      {path.name}")
        if len(files) > 6:
            print(f"      ... 其余 {len(files) - 6} 个")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="URDF/STL → 3 URDF + 4 MJCF 一键流水线。")
    parser.add_argument("--urdf", default=None, help="源 URDF（不在 Model/ 里会自动导入）")
    parser.add_argument("--robot", default=config.DEFAULT_ROBOT, help="机器人名（决定文件名）")
    parser.add_argument("--steps", nargs="*", default=list(DEFAULT_STEPS), choices=STEPS,
                        help=f"要跑的步骤，默认：{' '.join(DEFAULT_STEPS)}（obb 需显式指定）")
    parser.add_argument("--jobs", type=int, default=None,
                        help="并行数（不指定时：凸分解 1 个省内存，球体近似用 config.FOAM_JOBS）")
    parser.add_argument("--force", action="store_true", help="已有结果也重算")
    parser.add_argument("--max-faces", type=int, default=config.MAX_MESH_FACES,
                        help=f"简化面数上限（默认 {config.MAX_MESH_FACES}）")
    parser.add_argument("--keep-going", action="store_true", help="某一步失败也继续往下跑")
    parser.add_argument("--no-verify", action="store_true", help="跳过第 [6] 步体检")
    args = parser.parse_args()

    steps = [s for s in args.steps if not (args.no_verify and s == "verify")]
    started = time.time()
    config.ensure_dirs()

    # 找源 URDF：优先用参数给的，其次 Model/urdf/<robot>.urdf
    source = config.urdf_path(args.robot)
    if args.urdf:
        given = Path(args.urdf).expanduser().resolve()
        if not given.is_file():
            print(f"找不到 URDF：{given}")
            return 1
        if config.MODEL_DIR in given.parents:
            source = given                            # 已经在 Model 里，直接用
            print(f"源 URDF：{source.relative_to(config.MODEL_DIR)}")
        else:
            print(f"[0] 导入源模型 {given}")
            source = import_source_model(given)
    elif not source.is_file():
        print(f"找不到 {source}；用 --urdf 指定源 URDF 先导入一次")
        return 1
    else:
        print(f"源 URDF：{source.relative_to(config.MODEL_DIR)}")

    names = [p.stem for p in sorted(config.VISUAL_DIR.iterdir())
             if p.is_file() and not config.is_simplified(p)] if config.VISUAL_DIR.is_dir() else []
    robot = config.robot_name_from_urdf(source)
    print(f"机器人：{robot}，网格 {len(names)} 个：{', '.join(names)}")

    force = ["--force"] if args.force else []
    convex_jobs = args.jobs or 1                     # CoACD 吃内存，默认单进程
    sphere_jobs = args.jobs or config.FOAM_JOBS      # 球体近似是轻量进程，可以并行
    if "simplify" in steps:
        run_step("[1] 简化网格", "simplify_mesh.py",
                 ["--max-faces", str(args.max_faces)] + force, args.keep_going)
    if "convex" in steps:
        run_step("[2] 凸包分解（CoACD）", "decompose_convex.py",
                 ["--jobs", str(convex_jobs)] + force, args.keep_going)
    if "sphere" in steps:
        run_step("[3] 球体近似（foam CLI）", "decompose_sphere.py",
                 ["--quiet", "--jobs", str(sphere_jobs)] + force, args.keep_going)
    if "capsule" in steps:                               # 已从默认流程移除，需显式指定
        run_step("[3a] 胶囊近似（Pinocchio capsule-approximation，legacy）",
                 "decompose_capsule.py", ["--quiet"] + force, args.keep_going)
    if "obb" in steps:                                   # 已从默认流程移除，需显式指定
        run_step("[3b] OBB 计算（Open3D，legacy）", "decompose_obb.py", force, args.keep_going)
    if "urdf" in steps:
        run_step("[4] 生成 URDF 变体", "process_urdf.py", ["--urdf", str(source)],
                 args.keep_going)
    if "mjcf" in steps:
        run_step("[5] 转换 MJCF（4 套）", "urdf_to_mjcf.py", ["--all", "--robot", robot],
                 args.keep_going)
    if "verify" in steps:
        verify(robot)

    inventory(robot)
    print(f"\n流水线完成，用时 {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
