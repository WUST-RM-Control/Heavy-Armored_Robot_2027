"""生成 ``_convex.urdf`` / ``_sphere.urdf``（``_capsule.urdf`` / ``_obb.urdf`` 代码保留、默认不再生成）。

流程里的第 [4] 步。对原始 URDF 做三件事：

1. **视觉词条**：把原来指向 ``meshes/visual/xxx.STL`` 的路径改成相对 URDF 的
   相对路径（``../meshs/visual/xxx.STL``；如果这个网格超过面数上限、有简化副本，
   就指向 ``../meshs/simplified/xxx_simplified.STL``）。相对路径是这套东西能
   整体搬家的关键——不写任何绝对路径、不依赖 package://。
2. **碰撞词条**：删掉原来的图元碰撞（box/cylinder/sphere...），换成第 [2]/[3] 步
   分解出来的东西：
     * ``_convex.urdf`` ：每个连杆 N 条 <collision>，指向 ``meshs/colission/<stem>_<i>.obj``
     * ``_sphere.urdf`` ：每个连杆 M 条 <collision>，每条一个 ``<sphere radius>``
       图元 + 自己的 origin，球心/半径取 ``meshs/sphere/<stem>_sphere.json``
       （foam 生成的球体覆盖，取代了原来的胶囊近似）
     * ``_capsule.urdf``：每个连杆 1 条 <collision>，``<cylinder radius length>``
       图元（MuJoCo 载入后会变成 capsule），legacy
     * ``_obb.urdf``    ：每个连杆 1 条 <collision>，指向 ``meshs/obb/<stem>_obb.obj``
       （保留代码，默认不生成）
3. **原文件保留**：``urdf/<robot>.urdf`` 一个字都不改，各变体是另存的新文件。

用法::

    python scripts/process_urdf.py                       # 默认生成 convex + sphere
    python scripts/process_urdf.py --variant sphere
    python scripts/process_urdf.py --variant capsule     # legacy，需要先跑 decompose_capsule.py
    python scripts/process_urdf.py --variant obb         # legacy，需要先跑 decompose_obb.py
    python scripts/process_urdf.py --urdf urdf/wust_arm2.urdf --robot wust_arm2
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# --------------------------------------------------------------------------- #
# XML 小工具
# --------------------------------------------------------------------------- #
def read_urdf(path: Path) -> ET.ElementTree:
    """读 URDF，保留注释（SolidWorks 导出的头注释别丢）"""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def write_urdf(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ET.tostring(tree.getroot(), encoding="utf-8"))


def mesh_visuals(link: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    """link 里所有 <visual> 及其 <mesh>（只取网格型视觉）"""
    found = []
    for visual in link.findall("visual"):
        mesh = visual.find("./geometry/mesh")
        if mesh is not None and mesh.get("filename"):
            found.append((visual, mesh))
    return found


def collision_element(mesh_file: str, origin: ET.Element | None,
                      scale: str | None) -> ET.Element:
    """造一条 <collision>（origin 沿用视觉网格的，scale 也照抄）"""
    collision = ET.Element("collision")
    if origin is not None:
        copied = ET.SubElement(collision, "origin")
        copied.attrib.update(origin.attrib)
    geometry = ET.SubElement(collision, "geometry")
    mesh = ET.SubElement(geometry, "mesh", {"filename": mesh_file})
    if scale:
        mesh.set("scale", scale)
    return collision


def capsule_collision(capsule: dict) -> ET.Element:
    """造一条胶囊碰撞词条：``<cylinder radius length>`` + 胶囊自己的 origin。

    为什么用 ``<cylinder>``：URDF 里没有胶囊图元，而 MuJoCo 官方约定
    「URDF 的 cylinder → MJCF 的 capsule」，所以 ``<cylinder radius length>``
    载入 MuJoCo 后就是胶囊（``size = [radius, length/2]``），和 pinocchio 参考
    实现 ``parse_urdf`` 里的做法一致。

    origin 取 ``meshs/capsule/<stem>_capsule.json`` 里的 center + 轴线姿态；局部
    +Z 是胶囊轴，``length`` 是两球心距（不是总长，总长 = length + 2·radius）。
    """
    collision = ET.Element("collision")
    ET.SubElement(collision, "origin", {
        "xyz": " ".join(f"{v:.9g}" for v in capsule["center"]),
        "rpy": " ".join(f"{v:.9g}" for v in capsule["rpy"]),
    })
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "cylinder", {
        "radius": f"{capsule['radius']:.9g}",
        "length": f"{capsule['length']:.9g}",
    })
    return collision


def sphere_collisions(spheres: list[dict]) -> list[ET.Element]:
    """造球体碰撞词条：**一个球一条 <collision>**。

    URDF 没有「一组球」这种图元，只能一个球一个 ``<sphere radius>`` + 自己的
    ``<origin xyz>``（rpy 恒为 0，球是各向同性的）。MuJoCo 原生支持 URDF 的
    ``<sphere>``，载入后是 ``mjGEOM_SPHERE``、``size[0] = radius``，所以球体变体
    到 MJCF 不需要任何后处理。
    """
    elements = []
    for sphere in spheres:
        collision = ET.Element("collision")
        ET.SubElement(collision, "origin", {
            "xyz": " ".join(f"{v:.9g}" for v in sphere["origin"]),
            "rpy": "0 0 0",
        })
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "sphere", {"radius": f"{sphere['radius']:.9g}"})
        elements.append(collision)
    return elements


def read_spheres(stem: str) -> list[dict] | None:
    """读某个连杆的球体近似（没有就返回 None，调用方给出提示）"""
    path = config.sphere_path(stem)
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("spheres", [])


def insert_index_after_visuals(link: ET.Element) -> int:
    """碰撞词条插在最后一个视觉词条之后（URDF 里顺序无所谓，这样好读）"""
    index = 0
    for position, child in enumerate(link):
        if child.tag in ("visual", "collision"):
            index = position + 1
    return index


def read_capsule(stem: str) -> dict | None:
    """读某个连杆的胶囊参数（没有就返回 None，调用方给出提示）"""
    path = config.capsule_path(stem)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# 主体
# --------------------------------------------------------------------------- #
def process_link(link: ET.Element, variant: str) -> dict | None:
    """处理一个 <link>，返回统计信息（没有网格视觉的连杆返回 None）"""
    visuals = mesh_visuals(link)
    if not visuals:
        return None
    name = link.get("name", "?")
    first_visual, first_mesh = visuals[0]
    stem = config.stem_of(first_mesh.get("filename"))
    scale = first_mesh.get("scale")

    # --- 1. 视觉词条：改成相对路径，超面数的用简化副本 ---
    chosen = config.visual_mesh(stem)
    for _visual, mesh in visuals:
        mesh.set("filename", config.rel_to_urdf(chosen))

    # --- 2. 碰撞词条：先删旧的，再按变体插新的 ---
    removed = len(link.findall("collision"))
    for collision in list(link.findall("collision")):
        link.remove(collision)

    origin = first_visual.find("origin")
    index = insert_index_after_visuals(link)

    if variant == "convex" or (variant in ("capsule", "sphere")
                               and (config.capsule_skipped(name, stem)
                                    or config.sphere_skipped(name, stem))):
        # 被 CAPSULE_SKIP_LINKS / SPHERE_SKIP_LINKS 点名的连杆退回凸包
        files = [config.rel_to_urdf(p) for p in config.collision_parts(stem)]
        for offset, mesh_file in enumerate(files):
            link.insert(index + offset, collision_element(mesh_file, origin, scale))
        return dict(link=name, stem=stem, visual=config.rel_to_urdf(chosen),
                    removed=removed, added=len(files), scale=scale,
                    kind="mesh" if files else "empty")

    if variant == "sphere":
        spheres = read_spheres(stem)
        if not spheres:
            print(f"WARNING: {name} 没有找到球体近似结果，碰撞词条保持为空")
            return dict(link=name, stem=stem, visual=config.rel_to_urdf(chosen),
                        removed=removed, added=0, kind="empty")
        for offset, element in enumerate(sphere_collisions(spheres)):
            link.insert(index + offset, element)
        radii = [float(s["radius"]) for s in spheres]
        return dict(link=name, stem=stem, visual=config.rel_to_urdf(chosen),
                    removed=removed, added=len(spheres), kind="sphere",
                    nspheres=len(spheres), radius_min=min(radii), radius_max=max(radii))

    if variant == "capsule":
        capsule = read_capsule(stem)
        if capsule is None:
            print(f"WARNING: {name} 没有找到胶囊近似结果，碰撞词条保持为空")
            return dict(link=name, stem=stem, visual=config.rel_to_urdf(chosen),
                        removed=removed, added=0, kind="empty")
        link.insert(index, capsule_collision(capsule))
        return dict(link=name, stem=stem, visual=config.rel_to_urdf(chosen),
                    removed=removed, added=1, kind="capsule",
                    radius=capsule["radius"], length=capsule["length"])

    if variant == "obb":
        path = config.obb_path(stem)
        files = [config.rel_to_urdf(path)] if path.is_file() else []
        for offset, mesh_file in enumerate(files):
            link.insert(index + offset, collision_element(mesh_file, origin, scale))
        if not files:
            print(f"WARNING: {name} 没有找到 OBB 分解结果，碰撞词条保持为空")
        return dict(link=name, stem=stem, visual=config.rel_to_urdf(chosen),
                    removed=removed, added=len(files), scale=scale, kind="mesh")

    raise ValueError(f"未知的碰撞变体：{variant}")


def build_variant(source_urdf: Path, variant: str) -> tuple[Path, list[dict]]:
    """生成一个变体，返回 (输出路径, 每个连杆的统计)"""
    tree = read_urdf(source_urdf)
    root = tree.getroot()
    robot = root.get("name") or source_urdf.stem
    target = config.urdf_path(robot, variant)

    stats = []
    for link in root.findall("link"):
        info = process_link(link, variant)
        if info is None:
            print(f"NOTE: {link.get('name')} 没有网格视觉词条，跳过（碰撞词条不动）")
        else:
            stats.append(info)

    write_urdf(tree, target)
    return target, stats


def verify_urdf(path: Path) -> dict:
    """自查：引用的网格都在不在 + yourdfpy 能不能解析 + 统计碰撞词条"""
    root = ET.parse(path).getroot()
    missing, meshes, collisions, primitives = [], 0, 0, 0
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        meshes += 1
        if filename.startswith(("package://", "file://", "http")):
            missing.append(f"{filename}（不是相对路径）")
            continue
        resolved = (path.parent / filename).resolve()
        if not resolved.is_file():
            missing.append(f"{filename} → {resolved}")
    for collision in root.iter("collision"):
        collisions += 1
        geometry = collision.find("geometry")
        if geometry is not None and len(geometry) and geometry[0].tag != "mesh":
            primitives += 1

    parsed = None
    try:
        import yourdfpy

        model = yourdfpy.URDF.load(str(path), load_meshes=False)
        parsed = dict(links=len(model.link_map), joints=len(model.joint_map))
    except Exception as exc:  # 解析失败不影响文件本身，但要报出来
        missing.append(f"yourdfpy 解析失败：{exc}")

    return dict(meshes=meshes, collisions=collisions, primitives=primitives,
                missing=missing, parsed=parsed)


def process(source_urdf: Path, variants: list[str] | None = None) -> dict[str, Path]:
    """生成所有变体"""
    config.ensure_dirs()
    if not source_urdf.is_file():
        raise FileNotFoundError(f"找不到源 URDF：{source_urdf}")

    variants = variants or list(config.DEFAULT_VARIANTS)
    outputs: dict[str, Path] = {}
    for variant in variants:
        robot = config.robot_name_from_urdf(source_urdf)
        print(f"\n[{variant}] 源文件 {source_urdf.name} → {config.urdf_path(robot, variant).name}")
        target, stats = build_variant(source_urdf, variant)
        for info in stats:
            if info.get("kind") == "capsule":
                detail = f"胶囊 r={info['radius']:.4f} L={info['length']:.4f}"
            elif info.get("kind") == "sphere":
                detail = (f"球体 {info['nspheres']} 个 "
                          f"r={info['radius_min'] * 1000:.1f}~{info['radius_max'] * 1000:.1f}mm")
            else:
                detail = f"碰撞 {info['removed']} 条 → {info['added']} 条"
            print(f"  {info['link']:10s} 视觉 → {info['visual']:42s} {detail}")
        check = verify_urdf(target)
        print(f"  写入 {target.relative_to(config.MODEL_DIR)}："
              f"{check['meshes']} 条 mesh 引用、{check['collisions']} 条 collision"
              f"（其中图元 {check['primitives']} 条）"
              f"{'，link/joint=' + str(check['parsed']['links']) + '/' + str(check['parsed']['joints']) if check['parsed'] else ''}")
        if check["missing"]:
            for item in check["missing"]:
                print(f"  WARNING: {item}")
        outputs[variant] = target
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成 _convex.urdf / _sphere.urdf（_capsule / _obb 为 legacy）。")
    parser.add_argument("--urdf", default=None,
                        help=f"源 URDF（默认 {config.URDF_DIR / (config.DEFAULT_ROBOT + '.urdf')}）")
    parser.add_argument("--variant", nargs="*", default=None,
                        choices=list(config.COLLISION_VARIANTS) + [None],
                        help=f"只生成指定变体（默认 {' '.join(config.DEFAULT_VARIANTS)}）")
    args = parser.parse_args()

    source = Path(args.urdf).expanduser().resolve() if args.urdf \
        else config.urdf_path(config.DEFAULT_ROBOT)
    process(source, variants=args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
