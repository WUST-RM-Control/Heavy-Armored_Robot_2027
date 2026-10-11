"""执行器注入：给 MJCF 加 PD（位置伺服）或 MOTOR（力矩）。

两套执行器的用途不一样，所以两套都生成：

* **PD / position**：``ctrl`` 就是目标关节角，``kp/kv`` 在 ``config.PD_GAINS`` 里配。
  路径规划、轨迹跟踪这类算法用它验证最省事——不用自己写力矩环。
* **MOTOR / motor**：``ctrl`` 就是关节力矩，上下限取 URDF 的 ``effort``。
  动力学、辨识、力控用它，才是真实的力矩输入。

公共细节
--------
* 力矩都按 URDF 的 ``effort`` 限幅（``forcelimited`` + ``forcerange``），
  PD 也一样——不然位置伺服会凭空输出无穷大力矩，仿真好看但不真实；
* 会顺手删掉关节上的 ``actuatorfrcrange``：限幅统一交给执行器，
  两处同时限制会互相打架（关节级那个会把力矩卡在 effort 上，执行器再大也没用）；
* 重复调用是幂等的，已有的 ``<actuator>`` 会先清掉。

用法（一般由 urdf_to_mjcf.py 调用）::

    from add_actuators import add_actuators
    add_actuators(mjcf_root, "pd", source_urdf)
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

import config


# --------------------------------------------------------------------------- #
# URDF 关节信息
# --------------------------------------------------------------------------- #
def joint_limits(urdf_file: Path) -> dict[str, dict]:
    """读 URDF 里每个关节的 type / effort / velocity / range。

    返回 ``{joint: dict(type, effort, velocity, lower, upper)}``
    """
    root = ET.parse(urdf_file).getroot()
    limits: dict[str, dict] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        limit = joint.find("limit")
        info = dict(type=joint.get("type", "revolute"), effort=None, velocity=None,
                    lower=None, upper=None)
        if limit is not None:
            for key in ("effort", "velocity", "lower", "upper"):
                value = limit.get(key)
                if value is not None:
                    info[key] = float(value)
        limits[name] = info
    return limits


def actuated_joints(model, limits: dict[str, dict]) -> list[tuple[str, int, int]]:
    """MJCF 里可以加执行器的关节：``[(name, joint_id, dof_id)]``"""
    found = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        found.append((name, joint_id, int(model.jnt_dofadr[joint_id])))
    return found


# --------------------------------------------------------------------------- #
# 写 XML
# --------------------------------------------------------------------------- #
def _fresh_actuator_element(root: ET.Element) -> ET.Element:
    """拿到一个空的 <actuator>（顺便清掉旧的，保证幂等）"""
    for existing in list(root.findall("actuator")):
        root.remove(existing)
    actuator = ET.Element("actuator")
    root.append(actuator)
    return actuator


def _strip_joint_frcrange(root: ET.Element) -> int:
    """删掉关节级的 actuatorfrcrange（限幅只留在执行器上）"""
    removed = 0
    for joint in root.iter("joint"):
        if joint.get("actuatorfrcrange") is not None:
            joint.attrib.pop("actuatorfrcrange")
            removed += 1
    return removed


def add_motor_actuators(root: ET.Element, model, limits: dict[str, dict],
                        gear: float = config.MOTOR_GEAR,
                        effort_overrides: dict[str, float] | None = None) -> int:
    """MOTOR：``ctrl`` = 关节力矩"""
    actuator = _fresh_actuator_element(root)
    overrides = effort_overrides if effort_overrides is not None else config.MOTOR_EFFORT_OVERRIDES
    count = 0
    for name, _joint_id, _dof_id in actuated_joints(model, limits):
        effort = overrides.get(name, limits.get(name, {}).get("effort"))
        attrs = {"name": f"act_{name}", "joint": name, "gear": f"{gear:g}"}
        if effort:
            attrs.update({
                "ctrllimited": "true", "ctrlrange": f"{-effort:.12g} {effort:.12g}",
                "forcelimited": "true", "forcerange": f"{-effort:.12g} {effort:.12g}",
            })
        else:
            attrs["ctrllimited"] = "false"
            print(f"  WARNING: {name} 在 URDF 里没有 effort，力矩不限幅")
        ET.SubElement(actuator, "motor", attrs)
        count += 1
    stripped = _strip_joint_frcrange(root)
    print(f"  注入 {count} 个 motor 执行器"
          f"{f'，去掉 {stripped} 个关节级 actuatorfrcrange' if stripped else ''}")
    return count


def add_pd_actuators(root: ET.Element, model, limits: dict[str, dict],
                     default: dict | None = None,
                     gains: dict[str, dict] | None = None) -> int:
    """PD：``ctrl`` = 目标关节角（position 执行器 + kp/kv）"""
    actuator = _fresh_actuator_element(root)
    default = default or config.DEFAULT_PD
    gains = gains if gains is not None else config.PD_GAINS
    count = 0
    for name, _joint_id, _dof_id in actuated_joints(model, limits):
        info = limits.get(name, {})
        gain = dict(default)
        gain.update(gains.get(name, {}))
        attrs = {
            "name": f"act_{name}", "joint": name,
            "kp": f"{gain['kp']:g}", "kv": f"{gain['kv']:g}",
        }
        lower, upper = info.get("lower"), info.get("upper")
        if lower is not None and upper is not None:
            # ctrl 直接就是目标角，范围按关节行程给，超范围会被夹住
            attrs.update({"ctrllimited": "true", "ctrlrange": f"{lower:.12g} {upper:.12g}"})
        else:
            attrs["ctrllimited"] = "false"
        effort = info.get("effort")
        if effort:
            attrs.update({"forcelimited": "true", "forcerange": f"{-effort:.12g} {effort:.12g}"})
        ET.SubElement(actuator, "position", attrs)
        count += 1
    _strip_joint_frcrange(root)
    print(f"  注入 {count} 个 position(PD) 执行器"
          f"（kp/kv 来自 config.PD_GAINS，缺省 {default['kp']:g}/{default['kv']:g}）")
    return count


def add_actuators(root: ET.Element, actuator_type: str, urdf_file: Path, model) -> int:
    """统一入口：actuator_type ∈ {pd, motor}"""
    if actuator_type not in config.ACTUATOR_TYPES:
        raise ValueError(f"未知执行器类型：{actuator_type}（可选 {config.ACTUATOR_TYPES}）")
    limits = joint_limits(urdf_file)
    if actuator_type == "motor":
        return add_motor_actuators(root, model, limits)
    return add_pd_actuators(root, model, limits)
