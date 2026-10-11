"""接触排除注入：把相邻连杆之间的自碰撞关掉。

URDF 的语义里，父子连杆通过关节连在一起，本来就不该互相碰撞；MuJoCo 也有
``filterparent`` 兜底，但这里显式写成 ``<exclude>``，好处是：

* 一眼能看出哪些对不参与碰撞，换机器人时不会因为 filterparent 的行为差异踩坑；
* 根连杆被焊进 ``<worldbody>``（它的 geom 直接挂在 world 下），
  ``world ↔ 第一节`` 这一对必须显式排除，否则第一个关节球会一直顶着基座；
* ``config.EXTRA_EXCLUDES`` 里还能加手工排除对（例如某个连杆和腕部永远不碰）。

用法（一般由 urdf_to_mjcf.py 调用）::

    from add_contact_exclude import add_excludes_from_bodies
    add_excludes_from_bodies(mjcf_root)
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import config


def body_pairs(root: ET.Element) -> list[tuple[str, str]]:
    """从 MJCF 的 body 树里取出所有 (父, 子) 对。

    根连杆被 MuJoCo 焊进 world，它的 geom 在 ``<worldbody>`` 下；
    第一节 body 的父节点就是 world —— 正好对应我们要排除的那一对。
    """
    pairs: list[tuple[str, str]] = []

    def walk(element: ET.Element, parent: str) -> None:
        for body in element.findall("body"):
            name = body.get("name")
            if not name:
                continue
            pairs.append((parent, name))
            walk(body, name)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        walk(worldbody, "world")
    return pairs


def add_excludes(root: ET.Element, pairs: list[tuple[str, str]],
                 extra: list[tuple[str, str]] | None = None) -> int:
    """把 (父, 子) 对写进 ``<contact><exclude>``，返回新增条数"""
    body_names = {b.get("name") for b in root.iter("body")} | {"world"}
    contact = root.find("contact")
    if contact is None:
        contact = ET.Element("contact")
        worldbody = root.find("worldbody")
        position = list(root).index(worldbody) + 1 if worldbody is not None else len(root)
        root.insert(position, contact)

    existing = {(e.get("body1"), e.get("body2")) for e in contact.findall("exclude")}
    existing |= {(b, a) for a, b in existing}

    added = 0
    for body1, body2 in list(pairs) + list(extra or []):
        if body1 not in body_names or body2 not in body_names:
            print(f"  WARNING: exclude 里的 body 不存在：{body1} ↔ {body2}，跳过")
            continue
        if (body1, body2) in existing:
            continue
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
        existing.add((body1, body2))
        added += 1

    if added == 0 and len(contact) == 0:
        root.remove(contact)                    # 空 <contact> 不留
    return added


def add_excludes_from_bodies(root: ET.Element, extra: list[tuple[str, str]] | None = None,
                             enabled: bool = config.EXCLUDE_ADJACENT_LINKS) -> int:
    """按 body 树自动排除相邻连杆（+ config 里的额外排除对）"""
    return add_excludes(root, body_pairs(root) if enabled else [],
                        extra=extra if extra is not None else config.EXTRA_EXCLUDES)


def main() -> int:
    """单独跑：给一个 MJCF 补上接触排除（幂等）"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="给 MJCF 注入相邻连杆接触排除。")
    parser.add_argument("--xml", required=True, help="要处理的 MJCF")
    args = parser.parse_args()

    path = Path(args.xml).expanduser().resolve()
    tree = ET.parse(path)
    added = add_excludes_from_bodies(tree.getroot())
    ET.indent(tree, space="  ")
    path.write_bytes(ET.tostring(tree.getroot(), encoding="utf-8"))
    print(f"{path.name}: 新增 {added} 条 exclude")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
