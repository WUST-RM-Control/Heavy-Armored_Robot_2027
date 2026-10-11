"""URDF → MJCF 转换：转换 + 视觉网格 + 接触排除 + 执行器注入。

流程里的第 [5] 步，也是 ``_convex`` / ``_sphere`` 两套碰撞 × ``pd`` / ``motor``
两套执行器 = 四个 xml 的产地（``_capsule`` / ``_obb`` 那两套代码保留、默认不再生成）。

做五件事（照抄主工程 ``urdf/mjcf/urdf_to_mjcf.py`` 的思路，改成相对路径 + 模块化）：

1. **转换**：直接让 MuJoCo 自己读 URDF（``mj_saveLastXML``），这样关节轴向、
   惯性、关节限位这些细节和 MuJoCo 的官方实现保持一致；
2. **补视觉网格**：MuJoCo 只用 URDF 的 ``<collision>`` 生成 geom，``<visual>``
   会被丢掉。这里用 yourdfpy 读回视觉网格，登记进 ``<asset>``，并在对应 body 里
   加一个 ``class="visual"``（contype/conaffinity=0、density=0，只显示不碰撞）的 geom；
3. **胶囊图元（legacy）**：URDF 没有胶囊，``_capsule.urdf`` 里写的是 ``<cylinder>``，
   MuJoCo 会原样载成圆柱。这里按官方约定把 ``_capsule.urdf`` 来的圆柱 geom 改成
   capsule（``size=[radius, half_length]`` 不变，只是把平头换成球头），见
   :meth:`UrdfToMjcf.set_capsule_geoms`。
   **球体变体（``_sphere.urdf``）不需要任何后处理**：URDF 的 ``<sphere>`` 是标准
   图元，MuJoCo 直接载成 ``mjGEOM_SPHERE``（``size[0] = radius``）；
4. **接触排除**：见 ``add_contact_exclude.py``；
5. **执行器**：见 ``add_actuators.py``（pd / motor）。

另外会补关节默认动力学（damping/armature/frictionloss）和地面平面，
最后用 MuJoCo 再加载一遍做验证。

用法::

    python scripts/urdf_to_mjcf.py --urdf urdf/wust_arm2_convex.urdf --actuator pd
    python scripts/urdf_to_mjcf.py --urdf urdf/wust_arm2_sphere.urdf --actuator motor
    python scripts/urdf_to_mjcf.py --all          # 四套一起生成
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_actuators  # noqa: E402
import add_contact_exclude  # noqa: E402
import config  # noqa: E402


class UrdfToMjcf:
    """一次转换任务（一个 URDF → 一个 MJCF）"""

    def __init__(self, urdf_path: Path, out_path: Path, actuator: str = "pd",
                 capsules: bool | None = None):
        self.urdf_path = Path(urdf_path).resolve()
        self.out_path = Path(out_path).resolve()
        self.actuator = actuator
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.capsules = capsules                # None = 按文件名自动判断
        self.tree: ET.ElementTree | None = None
        self.root: ET.Element | None = None
        self._urdf = None
        self._mesh_cache: dict[str, Path | None] = {}

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def convert(self) -> Path:
        print(f"\n[{self.urdf_path.name} → {self.out_path.name}] 执行器={self.actuator}")

        # 1. MuJoCo 原生加载 URDF，再存成 MJCF
        model = mujoco.MjModel.from_xml_path(str(self.urdf_path))
        print(f"  MuJoCo 载入 URDF：nbody={model.nbody} njnt={model.njnt} "
              f"ngeom={model.ngeom} nu={model.nu}")
        mujoco.mj_saveLastXML(str(self.out_path), model)

        # 2. 读回 XML（保留注释）并修 mesh 路径
        self.read(self.out_path)
        fixed = self.fix_mesh_paths()

        # 3. 补视觉网格 / 圆柱→胶囊 / 关节动力学 / 接触排除 / 执行器 / 地面
        visuals = self.add_visual_geoms()
        capsule_geoms = self.set_capsule_geoms()
        dynamics = self.add_joint_dynamics()
        excludes = add_contact_exclude.add_excludes_from_bodies(self.root)
        actuators = add_actuators.add_actuators(self.root, self.actuator, self.urdf_path, model)
        collisions = self.set_collision_group()
        ground = self.add_ground_plane()

        self.write()
        print(f"  修补：mesh 路径 {fixed} 条、视觉 geom {visuals} 个、"
              f"胶囊图元 {capsule_geoms} 个、关节动力学 {dynamics} 组、"
              f"接触排除 {excludes} 条、执行器 {actuators} 个、"
              f"碰撞 geom 分组 {collisions} 个、地面 {ground} 个")

        # 4. 验证
        final = mujoco.MjModel.from_xml_path(str(self.out_path))
        print(f"  验证通过：nbody={final.nbody} njnt={final.njnt} ngeom={final.ngeom} "
              f"nmesh={final.nmesh} nu={final.nu}")
        return self.out_path

    # ------------------------------------------------------------------ #
    # XML 读写
    # ------------------------------------------------------------------ #
    def read(self, path: Path) -> ET.Element:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        self.tree = ET.parse(path, parser=parser)
        self.root = self.tree.getroot()
        return self.root

    def write(self, path: Path | None = None) -> Path:
        assert self.tree is not None, "先调用 read()"
        target = Path(path) if path is not None else self.out_path
        ET.indent(self.tree, space="  ")
        target.write_bytes(ET.tostring(self.root, encoding="utf-8"))
        return target

    # ------------------------------------------------------------------ #
    # 各步骤
    # ------------------------------------------------------------------ #
    def fix_mesh_paths(self) -> int:
        """MuJoCo 存的是绝对路径，改成相对 MJCF 目录的相对路径（方便搬家）"""
        fixed = 0
        for mesh in self.root.iter("mesh"):
            old = mesh.get("file")
            if not old:
                continue
            if os.path.isabs(old) or old.startswith(("package://", "file://")):
                source = self.resolve_mesh(old)
                if source is None:
                    print(f"  WARNING: 找不到网格 {old}，路径保持不变")
                    continue
                mesh.set("file", config.rel_to_mjcf(source))
                fixed += 1
        compiler = self.root.find("compiler")
        if compiler is not None and compiler.get("meshdir"):
            # 相对路径已经写全了，留着 meshdir 反而会让基准目录对不上
            compiler.attrib.pop("meshdir")
        return fixed

    def resolve_mesh(self, filename: str) -> Path | None:
        """把 URDF 里的 mesh 路径解析成磁盘文件（绝对路径 / package:// 都收）"""
        if filename in self._mesh_cache:
            return self._mesh_cache[filename]

        name = filename.replace("package://", "").replace("file://", "")
        candidates = [Path(name), self.urdf_path.parent / name,
                      config.URDF_DIR / name, config.MESHES_DIR / name]
        # package:// 大多指向没搬过来的 ROS 包目录，退化成按文件名在 Model 里找
        resolved = next((c for c in candidates if c.is_file()), None)
        if resolved is None:
            matches = sorted(config.MESHES_DIR.rglob(Path(name).name))
            resolved = matches[0] if matches else None
        self._mesh_cache[filename] = resolved
        return resolved

    def _ensure_visual_class(self) -> None:
        """<default><default class="visual">：只显示、不碰撞、不计质量"""
        default = self.root.find("default")
        if default is None:
            default = ET.Element("default")
            compiler = self.root.find("compiler")
            self.root.insert(1 if compiler is not None else 0, default)
        if default.find(f"default[@class='{config.VISUAL_CLASS}']") is None:
            node = ET.SubElement(default, "default", {"class": config.VISUAL_CLASS})
            ET.SubElement(node, "geom", {
                "type": "mesh",
                "contype": "0",
                "conaffinity": "0",
                "group": str(config.VISUAL_GROUP),
                "density": "0",
            })

    def add_visual_geoms(self) -> int:
        """把 URDF 的 <visual> 网格补成 MJCF 的视觉 geom"""
        self._ensure_visual_class()
        asset = self.root.find("asset")
        if asset is None:
            asset = ET.Element("asset")
            worldbody = self.root.find("worldbody")
            position = list(self.root).index(worldbody) if worldbody is not None else len(self.root)
            self.root.insert(position, asset)

        added = 0
        for link_name, link in self.urdf_model().link_map.items():
            body = self.body_element(link_name)
            if body is None:
                continue
            for index, visual in enumerate(link.visuals):
                geometry = visual.geometry
                if geometry is None or geometry.mesh is None:
                    continue
                added += self._add_visual_geom(body, asset, link_name, index, visual)
        return added

    def _add_visual_geom(self, body, asset, link_name: str, index: int, visual) -> int:
        mesh_name = f"{link_name}_visual" if index == 0 else f"{link_name}_visual_{index}"
        if any(m.get("name") == mesh_name for m in asset.findall("mesh")):
            return 0
        source = self.resolve_mesh(visual.geometry.mesh.filename)
        if source is None or not source.is_file():
            print(f"  WARNING: {link_name} 的视觉网格 {visual.geometry.mesh.filename} 找不到，跳过")
            return 0

        attrs = {"name": mesh_name, "file": config.rel_to_mjcf(source)}
        scale = visual.geometry.mesh.scale
        if scale is not None:
            scale = np.asarray(scale, dtype=float).ravel()
            if scale.size == 1:
                scale = np.repeat(scale, 3)
            if not np.allclose(scale, 1.0):
                attrs["scale"] = " ".join(f"{v:.6g}" for v in scale)
        ET.SubElement(asset, "mesh", attrs)

        geom_attrs = {"name": mesh_name, "class": config.VISUAL_CLASS, "mesh": mesh_name}
        geom_attrs.update(self._pose_attributes(visual.origin))
        rgba = self._rgba(visual)
        if rgba:
            geom_attrs["rgba"] = rgba

        position = self._geom_insert_index(body)
        body.insert(position, ET.Element("geom", geom_attrs))
        return 1

    def add_joint_dynamics(self) -> int:
        """给关节补 damping / armature / frictionloss（URDF 里没有 <dynamics>）"""
        default = self.root.find("default")
        if default is None:
            default = ET.Element("default")
            self.root.insert(0, default)
        if default.find("joint") is not None:
            return 0
        ET.SubElement(default, "joint", {
            "damping": f"{config.JOINT_DAMPING:g}",
            "armature": f"{config.JOINT_ARMATURE:g}",
            "frictionloss": f"{config.JOINT_FRICTIONLOSS:g}",
        })
        return 1

    def set_capsule_geoms(self) -> int:
        """把 ``_capsule.urdf`` 来的圆柱 geom 换成 capsule（平头 → 球头）。

        这是 **legacy 路径**：胶囊变体已经被 foam 球体近似取代，不过代码留着，
        ``--steps capsule`` 时还能用。

        URDF 没有胶囊图元，``_capsule.urdf`` 里写的是 ``<cylinder radius length>``，
        MuJoCo 会照原样载成 mjGEOM_CYLINDER。MuJoCo 官方约定是「URDF 的 cylinder
        对应 MJCF 的 capsule」，两者 ``size`` 都是 ``[radius, half_length]``，所以
        这里只改 ``type``，尺寸和位姿都不动。

        只在胶囊变体上启用（``--capsules auto`` 时按 ``_capsule`` 文件名判断），
        免得把原始 URDF 里真正的圆柱碰撞体也改掉。球体变体（``_sphere``）不走这里。
        """
        if self.capsules is None:
            self.capsules = "capsule" in self.urdf_path.stem.lower()
        if not self.capsules:
            return 0
        count = 0
        for geom in self.root.iter("geom"):
            if geom.get("type") == "cylinder":
                geom.set("type", "capsule")
                count += 1
        return count

    def set_collision_group(self) -> int:
        """把碰撞 geom 归到 group=3。

        MuJoCo viewer 默认只画 group 0~2，group 3 是关着的。这样打开模型看到的是
        干净的视觉网格（group 1），想检查碰撞体就按数字键 ``3`` 打开、按 ``1``
        关掉视觉——比较凸包 / OBB / 原始图元三套碰撞体时特别方便。
        """
        count = 0
        for parent in [self.root.find("worldbody"), *self.root.iter("body")]:
            if parent is None:
                continue
            for geom in parent.findall("geom"):
                if geom.get("class") == config.VISUAL_CLASS:
                    continue
                if geom.get("type") == "plane" or geom.get("name") == "ground_plane":
                    continue
                geom.set("group", str(config.COLLISION_GROUP))
                count += 1
        return count

    def add_ground_plane(self) -> int:
        """加地面（先检查，不重复加）"""
        if not config.ADD_GROUND_PLANE:
            return 0
        asset = self.root.find("asset")
        if asset is not None and not any(t.get("name") == "ground_plane_texture"
                                         for t in asset.findall("texture")):
            ET.SubElement(asset, "texture", {
                "name": "ground_plane_texture", "type": "2d", "builtin": "checker",
                "mark": "edge", "rgb1": "0.6 0.8 1.0", "rgb2": "0.6 0.95 0.6",
                "markrgb": "0.2 0.2 0.2", "width": "512", "height": "512",
            })
            ET.SubElement(asset, "material", {
                "name": "ground_plane_material", "texture": "ground_plane_texture",
                "texrepeat": "50 50", "reflectance": "0.1",
            })
        worldbody = self.root.find("worldbody")
        if worldbody is None:
            return 0
        if any(g.get("type") == "plane" for g in worldbody.findall("geom")):
            return 0
        ET.SubElement(worldbody, "geom", {
            "name": "ground_plane", "type": "plane",
            "pos": f"0 0 {config.GROUND_PLANE['z']:g}",
            "size": f"{config.GROUND_PLANE['size']:g} {config.GROUND_PLANE['size']:g} 0.1",
            "material": "ground_plane_material",
            "rgba": config.GROUND_PLANE["rgba"],
        })
        return 1

    # ------------------------------------------------------------------ #
    # 小工具
    # ------------------------------------------------------------------ #
    def urdf_model(self):
        """yourdfpy 只读一次（只要结构，不加载网格）"""
        if self._urdf is None:
            import yourdfpy

            self._urdf = yourdfpy.URDF.load(str(self.urdf_path), load_meshes=False)
        return self._urdf

    def body_element(self, link_name: str) -> ET.Element | None:
        """URDF link → MJCF 元素（根连杆的 geom 挂在 <worldbody> 下）"""
        if link_name == self.urdf_model().base_link:
            return self.root.find("worldbody")
        for body in self.root.iter("body"):
            if body.get("name") == link_name:
                return body
        return None

    @staticmethod
    def _geom_insert_index(body: ET.Element) -> int:
        """插到最后一个 geom 之后、子 body 之前（MJCF 要求子 body 排最后）"""
        index = 0
        for position, child in enumerate(body):
            if child.tag == "body":
                break
            index = position + 1
        return index

    @staticmethod
    def _pose_attributes(origin) -> dict:
        """URDF origin（4x4）→ MJCF 的 pos / quat（单位阵就省略）"""
        if origin is None:
            return {}
        origin = np.asarray(origin, dtype=float)
        attrs = {}
        if not np.allclose(origin[:3, 3], 0.0, atol=1e-12):
            attrs["pos"] = " ".join(f"{v:.6g}" for v in origin[:3, 3])
        if not np.allclose(origin[:3, :3], np.eye(3), atol=1e-9):
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, origin[:3, :3].flatten())
            if quat[0] < 0:
                quat = -quat
            attrs["quat"] = " ".join(f"{v:.6g}" for v in quat)
        return attrs

    @staticmethod
    def _rgba(visual) -> str | None:
        """URDF 的 <material><color rgba> → MJCF geom 的 rgba"""
        material = getattr(visual, "material", None)
        color = getattr(material, "color", None) if material is not None else None
        if color is None or getattr(color, "rgba", None) is None:
            return None
        return " ".join(f"{float(c):.6g}" for c in np.asarray(color.rgba).ravel())


def convert_all(robot: str = config.DEFAULT_ROBOT,
                variants: list[tuple[str, str]] | None = None,
                capsules: bool | None = None) -> dict[tuple[str, str], Path]:
    """把一个机器人的四套 MJCF 全生成出来"""
    variants = variants or config.MJCF_VARIANTS
    outputs: dict[tuple[str, str], Path] = {}
    for collision, actuator in variants:
        source = config.urdf_path(robot, collision)
        if not source.is_file():
            print(f"WARNING: 缺少 {source.name}，跳过 {collision}+{actuator}")
            continue
        target = config.mjcf_path(collision, actuator, robot)
        outputs[(collision, actuator)] = UrdfToMjcf(
            source, target, actuator, capsules=capsules).convert()
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="URDF → MJCF（含视觉网格/接触排除/执行器）。")
    parser.add_argument("--urdf", default=None, help="源 URDF（默认按 robot+collision 拼）")
    parser.add_argument("--out", default=None, help="输出 MJCF（默认 mjcf/<robot>_<碰撞>_<执行器>.xml）")
    parser.add_argument("--robot", default=config.DEFAULT_ROBOT)
    parser.add_argument("--collision", default="convex", choices=list(config.COLLISION_VARIANTS))
    parser.add_argument("--actuator", default="pd", choices=list(config.ACTUATOR_TYPES))
    parser.add_argument("--capsules", choices=("auto", "yes", "no"), default="auto",
                        help="圆柱 geom 是否转成 capsule（默认 auto：按 _capsule 文件名判断）")
    parser.add_argument("--all", action="store_true", help="四套一起生成")
    args = parser.parse_args()

    config.ensure_dirs()
    capsules = {"auto": None, "yes": True, "no": False}[args.capsules]
    if args.all:
        convert_all(args.robot, capsules=capsules)
        return 0

    source = Path(args.urdf).expanduser().resolve() if args.urdf \
        else config.urdf_path(args.robot, args.collision)
    target = Path(args.out).expanduser().resolve() if args.out \
        else config.mjcf_path(args.collision, args.actuator, args.robot)
    UrdfToMjcf(source, target, args.actuator, capsules=capsules).convert()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
