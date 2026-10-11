"""流水线统一配置：所有路径和参数都从这里取。

设计原则
--------
1. **全部相对路径**：``MODEL_DIR`` 由本文件位置反推（``Model/scripts/config.py``
   → ``Model/``），脚本之间传的都是相对于 Model 目录的路径。整个 Model 文件夹
   拷到别的机器/别的工程下面照样能跑，不依赖任何绝对路径。
2. **参数集中**：面数上限、CoACD 档位、foam 球体近似、PD/MOTOR 增益、地面、
   关节默认动力学都在这里，改参数不用翻脚本。
3. **命名约定集中**：视觉/简化/碰撞/Sphere（球体）/Capsule/OBB 的文件名规则只在
   这里定义一次，``simplify_mesh.py`` 写、``process_urdf.py`` 读，保证两边一致。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径（全部相对 Model 目录）
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = Path(__file__).resolve().parent          # Model/scripts
MODEL_DIR = SCRIPTS_DIR.parent                         # Model

MESHES_DIR = MODEL_DIR / "meshs"
VISUAL_DIR = MESHES_DIR / "visual"                     # 输入：原始 STL
SIMPLIFIED_DIR = MESHES_DIR / "simplified"             # 输出：简化后的 STL
COLLISION_DIR = MESHES_DIR / "colission"               # 输出：凸包部件 obj
SPHERE_DIR = MESHES_DIR / "sphere"                     # 输出：球体近似 json（foam）
CAPSULE_DIR = MESHES_DIR / "capsule"                   # 输出：胶囊近似 obj + json（保留，默认不再调用）
OBB_DIR = MESHES_DIR / "obb"                           # 输出：OBB 盒 obj（保留，默认不再调用）

URDF_DIR = MODEL_DIR / "urdf"                          # 原始 / _convex / _sphere(/ _capsule / _obb)
MJCF_DIR = MODEL_DIR / "mjcf"                          # 4 个 xml

#: 默认机器人（文件名去掉 .urdf）
DEFAULT_ROBOT = "robot"

#: 简化副本的后缀：LINK3.STL -> LINK3_simplified.STL
SIMPLIFIED_SUFFIX = "_simplified"
#: 凸包部件命名：LINK1_0.obj, LINK1_1.obj ...
CONVEX_PATTERN = "{stem}_{index}.obj"
#: 球体近似命名：LINK1_sphere.json（里面是一串 {origin, radius}）
SPHERE_PATTERN = "{stem}_sphere.json"
#: 胶囊近似命名：LINK1_capsule.obj（参数在同名 .json 里，legacy）
CAPSULE_PATTERN = "{stem}_capsule.obj"
#: OBB 盒命名：LINK1_obb.obj
OBB_PATTERN = "{stem}_obb.obj"

#: 网格后缀（大小写都收）
MESH_SUFFIXES = (".STL", ".stl", ".OBJ", ".obj", ".PLY", ".ply")

# --------------------------------------------------------------------------- #
# 简化
# --------------------------------------------------------------------------- #
#: 目标面数上限。MuJoCo 的 STL 解码硬上限是 20 万面，这里按课题要求压到 10 万
MAX_MESH_FACES = 100_000
#: MuJoCo 硬上限，超过就加载失败（只用于告警文案）
MUJOCO_FACE_LIMIT = 200_000

# --------------------------------------------------------------------------- #
# 凸包分解（CoACD）
# --------------------------------------------------------------------------- #
#: 档位照抄 urdf/meshes/coacd_meshs.py，另加一个 default
CONVEX_PROFILES: dict[str, dict] = {
    "finger": dict(
        threshold=0.03, max_convex_hull=32, preprocess_resolution=80,
        resolution=2500, mcts_nodes=30, mcts_iterations=300, mcts_max_depth=4,
        merge=False, decimate=True, max_ch_vertex=64,
    ),
    "peg": dict(
        threshold=0.01, max_convex_hull=64, preprocess_resolution=100,
        resolution=3000, mcts_nodes=40, mcts_iterations=500, mcts_max_depth=4,
        merge=False, decimate=True, max_ch_vertex=64,
    ),
    "link": dict(
        threshold=0.08, max_convex_hull=12, preprocess_resolution=50,
        resolution=1200, mcts_nodes=15, mcts_iterations=120, mcts_max_depth=3,
        merge=True, decimate=True, max_ch_vertex=32,
    ),
    "base": dict(
        threshold=0.12, max_convex_hull=6, preprocess_resolution=30,
        resolution=900, mcts_nodes=10, mcts_iterations=50, mcts_max_depth=2,
        merge=True, decimate=True, max_ch_vertex=16,
    ),
}
CONVEX_PROFILES["default"] = CONVEX_PROFILES["link"]
CONVEX_PROFILES["arm"] = CONVEX_PROFILES["link"]
CONVEX_PROFILES["gripper"] = CONVEX_PROFILES["finger"]
CONVEX_PROFILES["hole"] = CONVEX_PROFILES["peg"]
CONVEX_PROFILES["fixture"] = CONVEX_PROFILES["peg"]
CONVEX_PROFILES["mount"] = CONVEX_PROFILES["base"]

#: 连杆名 → 档位。按顺序匹配（不区分大小写、子串匹配），第一个命中的生效。
#: 换成别的机器人时只要在这里加规则，不用改脚本。
PART_PROFILE_RULES: list[tuple[str, str]] = [
    ("finger", "finger"),
    ("gripper", "gripper"),
    ("peg", "peg"),
    ("hole", "peg"),
    ("fixture", "peg"),
    ("base", "base"),
    ("mount", "base"),
    ("link", "link"),
    ("arm", "link"),
]

#: CoACD 通用参数
CONVEX_PREPROCESS_MODE = "auto"

# --------------------------------------------------------------------------- #
# 球体近似（foam）—— 取代原来的胶囊近似，成为默认的「轻量碰撞体」方案
# --------------------------------------------------------------------------- #
# 算法出处：https://github.com/CoMMALab/foam （论文 arXiv:2503.13704，
# *Foam: A Tool for Spherical Approximation of Robot Geometry*）。
# foam 不是自己实现算法，而是把它自带的 C++ 工具 **spheretree**（Trinity College
# Dublin, 2003 的球树构造库）串成一条流水线：
#
#   网格 → （不合法就 manifold 补洞 + 二次误差简化 + Humphrey 平滑）
#        → makeTree<Method> 造球树 → 取某一层球作为近似
#
# 五种方法（``FOAM_METHOD``）：
#   * ``medial``  中轴（medial axis）法，最贴合、最慢，foam 的默认
#   * ``grid``    网格法，快、偏保守
#   * ``spawn``   球体生长（spawn）法
#   * ``octree``  八叉树法，最快、最胖
#   * ``hubbard`` Hubbard 中轴近似，速度/贴合度居中
#
# **调用方式**：``decompose_sphere.py`` 不 import foam、不重写算法，而是直接把
# foam 自带的两个 CLI 脚本当子进程调起来（见 ``FOAM_CLI_MESH`` /
# ``FOAM_CLI_URDF``）——保证结果和官方脚本一字不差。
#
# 环境是两级的（因为 foam 需要 fire/xmltodict/vhacdx，而主环境 ~/mujoco_env 没有）：
#   * ``decompose_sphere.py`` 自己只依赖 numpy/trimesh，跑在 ~/mujoco_env；
#   * foam 的 CLI 用 ``FOAM_PYTHON`` 指定的解释器跑，默认自动探测
#     ``<工程>/.pixi/envs/default/bin/python``（pixi.toml 里已经声明了 foam）。
#   * C++ 二进制（makeTreeMedial 等）不在 pip 包里，需要编译一次：
#     ``python scripts/setup_foam.py``。

#: 跑 foam CLI 的解释器。空 = 自动探测（--python > config > sys.executable
#: > <工程>/.pixi/envs/default/bin/python > PATH 里的 python3）
FOAM_PYTHON = ""
#: foam CLI 脚本名（相对 foam 包旁边的 scripts/ 目录）
FOAM_CLI_MESH = "generate_spheres.py"
FOAM_CLI_URDF = "generate_sphere_urdf.py"
#: 单次 CLI 调用超时（秒）——medial 法在大网格上会跑几分钟
FOAM_TIMEOUT = 3600

#: 球体生成算法：medial / grid / spawn / octree / hubbard
FOAM_METHOD = "medial"
#: 球树层数：depth=d 时一共 d+1 层，最深层最多 branch**d 个球（取最深层做近似）
#: 实测（LINK1，38k 面）：d=1 → 8 球/93s 贴合度一般；d=2 → 62 球/93s 明显变好；
#: d=3 → 408 球/202s 和中档几乎一样（球多 6 倍但贴合度没提升），所以默认 d=2。
FOAM_DEPTH = 2
#: 分支因子（每层分裂成几个子球）
FOAM_BRANCH = 8
#: 网格预处理：manifold 的目标叶子数 / 简化比例
FOAM_MANIFOLD_LEAVES = 1000
FOAM_SIMPLIFY_RATIO = 0.2
#: 让 spheretree 在真正生成球树前再校验一遍模型。**默认关**：
#: foam 内部已经用 `check_valid_for_spherization` 校验过一遍（不合法会先
#: manifold 补洞），最后这次校验既慢、又会让 depth>=3 的球树中途报
#: COVERAGE NOT VALID 直接失败，属于纯负担。
FOAM_VERIFY = False
#: 下面这些是 spheretree 的原生参数，照抄 foam 的默认值，一般不用动
FOAM_TESTER_LEVELS = 2       # 求球与模型的误差时，球面用几层采样点表示
FOAM_NUM_COVER = 5000        # 覆盖模型用的采样点数
FOAM_MIN_COVER = 5           # 每个三角形最少采样点数
FOAM_INIT_SPHERES = 1000     # 中轴近似的初始球数（medial）
FOAM_MIN_SPHERES = 200       # 每个子区域最少球数（medial）
FOAM_ER_FACT = 2             # 中轴细化时的误差收缩因子（medial）
FOAM_EXPAND = True           # EXPAND 优化（medial）
FOAM_MERGE = True            # MERGE 优化（medial）
FOAM_BURST = False           # BURST 优化（medial）
FOAM_OPTIMISE = True         # 非 medial 方法用 simplex 再优化一轮
FOAM_MAX_OPT_LEVEL = 1       # 优化到第几层（0 = 只优化第一层）
FOAM_BAL_EXCESS = 0.05       # BALANCE 优化允许的额外误差
FOAM_NUM_SAMPLES = 500       # Hubbard 法的采样点数
FOAM_MIN_SAMPLES = 1         # Hubbard 法每个三角形的最少采样点数
#: URDF 模式（--mode urdf）专用
FOAM_SHRINKAGE = 1.0         # 碰撞体整体缩放（<1 相当于收紧）
FOAM_THREADS = 8             # generate_sphere_urdf.py 的并行线程数

#: 同时跑几个连杆的 CLI（每个连杆一个 makeTree 进程，进程间互不影响）
FOAM_JOBS = 4
#: 半径余量（米）：所有球半径统一加这么多，抵消采样/浮点误差
SPHERE_MARGIN = 0.0005
#: 复验后把「没盖住的网格表面点」按最近的球分摊补半径（严格保证 网格 ⊆ 球并集）。
#: 关掉的话漏的地方就是碰撞盲区（json 里的 verify.penetration_after 会报出来）。
SPHERE_COMPENSATE = True
#: 半径下限（米）：0 = 不设下限（foam 会丢掉半径 <= 0 的球）
SPHERE_MIN_RADIUS = 0.0
#: 取球树的第几层：-1 = 最深层（球最多、最贴合），0 = 根层（1 个球/连杆）
SPHERE_LEVEL = -1
#: 丢掉「完全被另一个球包住」的球（无损剪枝：不改变并集，只是少几个 geom）
SPHERE_PRUNE_CONTAINED = True
SPHERE_PRUNE_EPS = 1e-9
#: 每个连杆最多保留多少个球（0 = 不限）。按「体积贡献」从大到小留
SPHERE_MAX_SPHERES = 0
#: 复验：用网格表面点检查「网格 ⊆ 球的并集」，没盖住就整体放大半径补齐
SPHERE_VERIFY = True
#: 复验采样上限（网格表面点；超过就随机抽）
SPHERE_VERIFY_SAMPLES = 200_000
#: 写出参数 json
SPHERE_WRITE_PARAMS = True
SPHERE_PARAMS_FILE = "sphere_params.json"
#: 顺带把球体并集导出成 obj（默认关：一个连杆几百个球，画出来是几十万面）
SPHERE_WRITE_MESH = False
SPHERE_MESH_SUBDIVISIONS = 1

#: 想让个别连杆不做球体近似、退回 CoACD 凸包，就把名字填进来（大小写不敏感）。
#: 典型场景：底盘这种方块大件，球体近似会糊住整个工作空间。
#:
#: ``base_link`` 是实测填进来的：底盘是 580×590×228mm 的方块，64 个球里最大的
#: 半径 149.5mm，球面比零件表面平均胖 85.5mm（最大 165mm），而它离 LINK2 的真实
#: 间隙只有 38.75mm —— 零位一上来就报 447 个假接触、JOINT1 被顶到 0.51rad。
#: 退回凸包后零位接触回到真值。
SPHERE_SKIP_LINKS: list[str] = ["base_link"]

# --------------------------------------------------------------------------- #
# 胶囊 / 椭圆近似（Pinocchio capsule-approximation 算法）—— legacy
# --------------------------------------------------------------------------- #
#: 已被 foam 球体近似取代（``decompose_sphere.py``）；``decompose_capsule.py``
#: 代码保留、默认不再调用。要重新生成：``run_pipeline.py --steps capsule urdf mjcf``。
#:
#: 算法出处：pinocchio 官方 examples/capsule-approximation.py
#: （Antonio El Khoury 博士论文 3.8.1「Computing minimum bounding capsules」）
#: pinocchio 4.x 已经不再导出 capsule_approximation/approximate_mesh 这两个
#: python 绑定，所以 decompose_capsule.py 里按同一份参考实现复刻：hppfcl 读网格
#: → 取凸包顶点 → PCA 初值 → scipy 约束优化最小体积胶囊。
#:
#: 胶囊 = 线段 (a, b) + 半径 r。最小体积胶囊同时是一个**最小外接**胶囊，所以
#: 它把零件整个包住：不会漏碰（安全），代价是细长连杆会偏胖（保守）。
#: 半径/线段端点都存进 meshs/capsule/capsule_params.json，想换成别的方法直接取用。

#: 收紧半径的余量（米）：最终半径取 ``拟合值 + margin``，保证胶囊严格包住网格
CAPSULE_MARGIN = 0.0005
#: 半径下限（米）：避免退化成一个点球
CAPSULE_MIN_RADIUS = 0.002
#: SLSQP 迭代上限 / 收敛容差
CAPSULE_MAX_ITER = 400
CAPSULE_FTOL = 1e-10
#: 写出参数 json
CAPSULE_WRITE_PARAMS = True
CAPSULE_PARAMS_FILE = "capsule_params.json"
#: 顺带导出胶囊网格 obj（形状和 URDF/MJCF 里的胶囊图元完全一致，便于对照看图）
CAPSULE_WRITE_MESH = True

#: 想让个别连杆不做胶囊、退回 CoACD 凸包，就把连杆名填进来（大小写不敏感）。
#: 非空时 ``_capsule.urdf`` 里这些连杆用凸包碰撞，属性和 ``_convex.urdf`` 一致。
CAPSULE_SKIP_LINKS: list[str] = []

# --------------------------------------------------------------------------- #
# OBB（Open3D）—— 保留代码，默认不再调用
# --------------------------------------------------------------------------- #
#: OBB 盒每个方向外扩的余量（米）。OBB 是顶点包围盒，理论上刚好包住，
#: 留一点点余量可以避免面片正好贴在盒面上时的抖动
OBB_MARGIN = 0.0002
#: 写出 OBB 参数的 json（方便和 box 图元方案对照）
OBB_WRITE_PARAMS = True
OBB_PARAMS_FILE = "obb_params.json"
#: 已经被 foam 球体近似取代的两套老方案：代码留着、默认不跑，
#: 想跑就 ``run_pipeline.py --steps capsule obb`` 显式点名
LEGACY_VARIANTS: list[str] = ["capsule", "obb"]

# --------------------------------------------------------------------------- #
# URDF → MJCF
# --------------------------------------------------------------------------- #
#: 视觉 geom 用的 default class（只看不碰撞、不贡献质量）
VISUAL_CLASS = "visual"
VISUAL_GROUP = 1
#: 碰撞 geom 的 group（viewer 里按数字键切换）
COLLISION_GROUP = 3

#: 关节默认动力学，抄主工程 wust_rm.xml
JOINT_DAMPING = 1.0
JOINT_ARMATURE = 0.01
JOINT_FRICTIONLOSS = 0.05

#: 相邻连杆接触排除（URDF 语义：父子连杆本来就不该互撞）
EXCLUDE_ADJACENT_LINKS = True
#: 额外要排除的 body 对，例如 [("LINK2", "LINK4")]
EXTRA_EXCLUDES: list[tuple[str, str]] = []

#: 地面
ADD_GROUND_PLANE = True
GROUND_PLANE = dict(
    size=10.0,
    z=-0.1,
    rgba="0.8 0.8 0.8 1.0",
)

# --------------------------------------------------------------------------- #
# 执行器
# --------------------------------------------------------------------------- #
#: PD（位置伺服）：用于路径规划/运动学验证，ctrl 就是目标关节角
#: kp/kv 按关节给，名字对不上就用 DEFAULT_PD
DEFAULT_PD = dict(kp=300.0, kv=30.0)
PD_GAINS: dict[str, dict[str, float]] = {
    # 大臂/基座惯量大，刚度给高一点；腕部小惯量给小一点，避免抖
    "JOINT1": dict(kp=400.0, kv=40.0),
    "JOINT2": dict(kp=500.0, kv=50.0),
    "JOINT3": dict(kp=400.0, kv=40.0),
    "JOINT4": dict(kp=200.0, kv=20.0),
    "JOINT5": dict(kp=200.0, kv=20.0),
    "JOINT6": dict(kp=100.0, kv=10.0),
    "JOINT7": dict(kp=100.0, kv=10.0),
}

#: MOTOR（力矩）：用于动力学验证，ctrl 就是关节力矩
MOTOR_GEAR = 1.0
#: 力矩上限默认取 URDF 的 effort；这里可以按关节覆盖
MOTOR_EFFORT_OVERRIDES: dict[str, float] = {}

#: 需要生成的 MJCF：碰撞模型 × 执行器。当前 = convex / sphere × pd / motor
MJCF_VARIANTS = [
    ("convex", "pd"),
    ("convex", "motor"),
    ("sphere", "pd"),
    ("sphere", "motor"),
]
#: 碰撞模型 → URDF 后缀
COLLISION_VARIANTS = {"convex": "_convex", "sphere": "_sphere",
                      "capsule": "_capsule", "obb": "_obb"}
#: 执行器类型
ACTUATOR_TYPES = ("pd", "motor")
#: 默认参与流水线的碰撞变体（capsule / obb 保留代码但不再调用）
DEFAULT_VARIANTS = ["convex", "sphere"]


# --------------------------------------------------------------------------- #
# 命名与路径工具
# --------------------------------------------------------------------------- #
def convex_name(stem: str, index: int) -> str:
    return CONVEX_PATTERN.format(stem=stem, index=index)


def capsule_name(stem: str) -> str:
    return CAPSULE_PATTERN.format(stem=stem)


def sphere_name(stem: str) -> str:
    return SPHERE_PATTERN.format(stem=stem)


def obb_name(stem: str) -> str:
    return OBB_PATTERN.format(stem=stem)


def simplified_path(stem: str) -> Path:
    """简化副本的完整路径（不管存不存在）"""
    return SIMPLIFIED_DIR / f"{stem}{SIMPLIFIED_SUFFIX}.STL"


def visual_path(stem: str) -> Path:
    """原始视觉网格：在 visual/ 下按名字找，后缀不限"""
    for suffix in MESH_SUFFIXES:
        candidate = VISUAL_DIR / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return VISUAL_DIR / f"{stem}.STL"


def visual_mesh(stem: str) -> Path:
    """URDF 视觉网格该用哪个文件：有简化副本就用简化副本，否则用原件。

    超过面数上限的原始 STL 直接塞进 MuJoCo 会解码失败，所以简化副本优先。
    """
    candidate = simplified_path(stem)
    return candidate if candidate.is_file() else visual_path(stem)


def collision_parts(stem: str) -> list[Path]:
    """某个连杆的凸包部件（按序号排序）"""
    if not COLLISION_DIR.is_dir():
        return []
    def key(path: Path) -> int:
        tail = path.stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else -1
    parts = [p for p in COLLISION_DIR.glob(f"{stem}_*") if p.suffix.lower() == ".obj"]
    return sorted(parts, key=key)


def capsule_path(stem: str) -> Path:
    """胶囊近似的参数 json（``meshs/capsule/<stem>_capsule.json``，legacy）"""
    return CAPSULE_DIR / (Path(capsule_name(stem)).with_suffix(".json").name)


def sphere_path(stem: str) -> Path:
    """球体近似的参数 json（``meshs/sphere/<stem>_sphere.json``）"""
    return SPHERE_DIR / sphere_name(stem)


def sphere_params_path() -> Path:
    """球体近似的汇总参数（``meshs/sphere/sphere_params.json``）"""
    return SPHERE_DIR / SPHERE_PARAMS_FILE


def capsule_mesh_path(stem: str) -> Path:
    """胶囊近似的网格（``meshs/capsule/<stem>_capsule.obj``，可选产物）"""
    return CAPSULE_DIR / capsule_name(stem)


def obb_path(stem: str) -> Path:
    return OBB_DIR / obb_name(stem)


def urdf_path(robot: str = DEFAULT_ROBOT, variant: str = "") -> Path:
    """variant: ""（原始）/ "convex" / "capsule" / "obb" """
    suffix = COLLISION_VARIANTS.get(variant, "")
    return URDF_DIR / f"{robot}{suffix}.urdf"


def mjcf_path(collision: str, actuator: str, robot: str = DEFAULT_ROBOT) -> Path:
    return MJCF_DIR / f"{robot}_{collision}_{actuator}.xml"


def rel_to_urdf(path: Path) -> str:
    """网格路径 → URDF 里用的相对路径（相对 urdf/ 目录）。

    URDF 里必须是正斜杠，在 Windows 上跑也不会写成反斜杠。
    """
    return os.path.relpath(Path(path).resolve(), URDF_DIR.resolve()).replace("\\", "/")


def rel_to_mjcf(path: Path) -> str:
    """网格路径 → MJCF 里用的相对路径（相对 mjcf/ 目录）"""
    return os.path.relpath(Path(path).resolve(), MJCF_DIR.resolve()).replace("\\", "/")


def stem_of(mesh_filename: str) -> str:
    """URDF 里的 mesh 文件名 → 连杆网格的 stem（去掉目录和后缀）"""
    return Path(str(mesh_filename).replace("\\", "/")).stem


def profile_for(link_or_mesh: str) -> str:
    """连杆名/网格名 → CoACD 档位"""
    name = str(link_or_mesh).lower()
    for keyword, profile in PART_PROFILE_RULES:
        if keyword in name:
            return profile
    return "default"


def capsule_skipped(*names: str) -> bool:
    """这些名字（连杆名 / 网格 stem）是否在 CAPSULE_SKIP_LINKS 里（大小写不敏感）"""
    if not CAPSULE_SKIP_LINKS:
        return False
    wanted = {str(n).lower() for n in names if n}
    return bool(wanted & {s.lower() for s in CAPSULE_SKIP_LINKS})


def sphere_skipped(*names: str) -> bool:
    """这些名字（连杆名 / 网格 stem）是否在 SPHERE_SKIP_LINKS 里（大小写不敏感）"""
    if not SPHERE_SKIP_LINKS:
        return False
    wanted = {str(n).lower() for n in names if n}
    return bool(wanted & {s.lower() for s in SPHERE_SKIP_LINKS})


def ensure_dirs() -> None:
    """把流水线需要的目录建出来（相对 Model 目录，随拷随用）"""
    for directory in (VISUAL_DIR, SIMPLIFIED_DIR, COLLISION_DIR, SPHERE_DIR, CAPSULE_DIR,
                      OBB_DIR, URDF_DIR, MJCF_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def robot_name_from_urdf(urdf_file: Path) -> str:
    """读 URDF 的 <robot name="...">"""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_file).getroot()
    return root.get("name") or urdf_file.stem


def is_simplified(path: Path) -> bool:
    return path.stem.endswith(SIMPLIFIED_SUFFIX)


def clean_stem(stem: str) -> str:
    """把 _simplified 之类派生后缀去掉，拿到原始 stem"""
    return re.sub(rf"{SIMPLIFIED_SUFFIX}$", "", stem, flags=re.IGNORECASE)
