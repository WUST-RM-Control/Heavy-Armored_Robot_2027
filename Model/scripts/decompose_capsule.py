"""胶囊（椭圆）近似：用 Pinocchio 的 capsule-approximation 算法给每个连杆拟合胶囊。

流程里的第 [3] 步，输入同凸分解（简化后的 STL），**取代原来的 OBB 近似**
（``decompose_obb.py`` 代码保留、默认不再调用）。

算法出处
--------
pinocchio 官方示例 ``examples/capsule-approximation.py``，实现的是
Antonio El Khoury 博士论文 3.8.1 节 *Computing minimum bounding capsules*：

1. ``approximate_mesh(filename, lMg)``：用 ``hppfcl.MeshLoader`` 读网格、把顶点
   变换到连杆坐标系，再交给 ``capsule_approximation``；
2. ``capsule_approximation(vertices)``：

   * ``pca_approximation``：顶点经 SVD 求主轴 → 主轴方向作线段、其余方向最大
     距离作半径，得到初值 ``(a0, b0, r0)``；
   * 以**胶囊体积**为目标、以「所有顶点到线段的距离 ≤ 半径」为约束做 SLSQP
     优化，初值半径再按 ``CONSTRAINT_INFLATION_RATIO`` 收紧一点，避免初值不可行。

   胶囊 = 线段 ``(a, b)`` + 半径 ``r``，最小体积胶囊同时也是**最小外接**胶囊，
   所以零件被完整包住：不会漏碰；代价是细长连杆偏胖（保守）。

pinocchio 4.1 的 python 绑定里已经没有 ``capsule_approximation`` /
``approximate_mesh`` 这两个符号（只剩 C++ 侧和示例脚本），所以这里按同一份参考
实现复刻，函数名保持一致：:func:`approximate_mesh` / :func:`capsule_approximation`
/ :func:`pca_approximation` / :func:`capsule_volume` / :func:`distance_points_segment`。

和参考实现的两处工程化改动（都在 json 里记录）：

* **拟合只用凸包顶点**。参考实现把全部顶点丢进优化，一个 18 万面的 STL 有 50 万
  个顶点，SLSQP 每次迭代都要遍历一遍，慢且没必要——最大距离必然出现在凸包顶点
  上，凸包把顶点数从 50 万降到 200 左右，结果等价。
* **半径取 ``拟合值 + CAPSULE_MARGIN``**，并且用**全部**顶点复验一遍
  （``max_surface_distance``），保证胶囊严格包住网格，不受浮点误差和
  凸包/渲染网格解码差异的影响。

输出::

    meshs/capsule/LINK1_capsule.json   # a / b / r、位姿、贴合度、复验结果
    meshs/capsule/LINK1_capsule.obj    # 形状与 URDF/MJCF 里胶囊图元一致的网格（可选）
    meshs/capsule/capsule_params.json  # 汇总

用法::

    python scripts/decompose_capsule.py
    python scripts/decompose_capsule.py --only LINK4 --margin 0.001
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

EPSILON = 1e-8
#: 关掉「SLSQP 未收敛」之类的中途提示（run_pipeline 里默认打开，输出干净些）
QUIET = False


# --------------------------------------------------------------------------- #
# 参考实现里的五个函数（名字、公式保持一致）
# --------------------------------------------------------------------------- #
def capsule_volume(a: np.ndarray, b: np.ndarray, r: float) -> float:
    """胶囊体积：圆柱段 + 两个半球"""
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)) * np.pi * r ** 2
                 + 4.0 / 3.0 * np.pi * r ** 3)


def distance_points_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """每个点到线段 ab 的距离（返回数组；``.max()`` 就是参考实现里的那个距离）"""
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-15:                                  # 退化成点，距离就是到该点的距离
        return np.linalg.norm(np.asarray(points, dtype=float) - np.asarray(a, dtype=float),
                              axis=1)
    direction = delta / norm
    relative = np.asarray(points, dtype=float) - np.asarray(a, dtype=float)
    t = np.clip(relative @ direction, 0.0, norm)        # 投影参数，夹在线段两端
    closest = np.asarray(a, dtype=float) + t[:, None] * direction
    return np.linalg.norm(np.asarray(points, dtype=float) - closest, axis=1)


def _capsule_gradient(points: np.ndarray, a: np.ndarray, b: np.ndarray,
                      index: int | None = None) -> tuple[float, np.ndarray, int]:
    """胶囊半径 ``r(a, b) = max_v dist(v, ab)`` 及其**次梯度**。

    返回 ``(r, [∂r/∂a(3), ∂r/∂b(3)], 支撑点序号)``：

    * 支撑点投影落在线段内部时，``∂r/∂a = −u``、``∂r/∂b = +u``（u 为线段单位向量），
      也就是把线段往两侧撑开；
    * 投影落在端点上时，退化成「点到球心」，``∂r/∂a = −(v−a)/‖v−a‖``。

    给 SLSQP 用解析梯度是必需的：目标函数里带 ``√(≈1e-8)``，数值差分会被放大成
    1e8 量级，优化器会直接崩到 r≈0 的退化解。
    """
    points = np.asarray(points, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    delta = b - a
    norm = float(np.linalg.norm(delta))

    if index is None:
        index = int(np.argmax(distance_points_segment(points, a, b)))
    vertex = points[index]
    gradient = np.zeros(6)

    if norm <= 1e-15:                                   # 线段退化成点
        offset = vertex - a
        distance = float(np.linalg.norm(offset))
        if distance > 0:
            gradient[:3] = -offset / distance
            gradient[3:] = offset / distance
        return distance, gradient, index

    direction = delta / norm
    along = float((vertex - a) @ direction)
    if along <= 0.0:                                    # 支撑点在 a 端
        offset = vertex - a
        distance = float(np.linalg.norm(offset))
        if distance > 0:
            gradient[:3] = -offset / distance
    elif along >= norm:                                 # 支撑点在 b 端
        offset = vertex - b
        distance = float(np.linalg.norm(offset))
        if distance > 0:
            gradient[3:] = offset / distance
    else:                                               # 支撑点在线段内部
        closest = a + along * direction
        perpendicular = vertex - closest
        distance = float(np.linalg.norm(perpendicular))
        if distance > 0:
            unit = perpendicular / distance
            gradient[:3] = -unit
            gradient[3:] = unit
    return distance, gradient, index


def feasible_capsule(vertices: np.ndarray, a: np.ndarray, b: np.ndarray,
                     iterations: int | None = None, tolerance: float = 1e-12
                     ) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """次梯度下降：直接最小化 ``r(a, b) = max_v dist(v, ab)``。

    和 :func:`capsule_approximation`（照抄参考实现的 SLSQP 形式）互补——那条路是
    不定方程式的优化，这里是最小化「最粗处」的确定性子问题，两个都不成问题时用
    体积更小的那个。参数化不同但目标一致：最小外接胶囊 = 最小体积胶囊。

    每步都做回溯线搜索，半径单调不增，所以返回的胶囊必定可行（不漏碰）。
    """
    points = np.asarray(vertices, dtype=float)
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    limit = config.CAPSULE_MAX_ITER if iterations is None else iterations
    step = 0.25 * max(float(np.linalg.norm(b - a)), float(np.abs(points).max()), 1e-6)
    radius = float(distance_points_segment(points, a, b).max())

    for _ in range(limit):
        _, gradient, _ = _capsule_gradient(points, a, b)
        magnitude = float(np.linalg.norm(gradient))
        if magnitude <= 1e-15:
            break
        direction = gradient / magnitude
        shrink = 1.0
        moved = False
        for _ in range(48):
            candidate_a = a - step * direction[:3]
            candidate_b = b - step * direction[3:]
            candidate = float(distance_points_segment(points, candidate_a, candidate_b).max())
            if candidate < radius - 1e-15:
                a, b, radius = candidate_a, candidate_b, candidate
                moved = True
                break
            step *= 0.5 * shrink
            shrink *= 0.5 if shrink > 1e-3 else 1.0
        if not moved:
            break
        step *= 1.6                                     # 走通了就放大步长，加快收敛

    return a, b, radius, bool(radius <= float(distance_points_segment(points, a, b).max())
                              + tolerance)


def pca_approximation(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA 初值：主轴定线段方向，其余方向最大距离定半径"""
    vertices = np.asarray(vertices, dtype=float)
    mean = vertices.mean(axis=0)
    centered = vertices - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=True)
    components = vh
    projected = centered @ components.T

    a0 = mean + components[0] * (projected[:, 0].min() - EPSILON)
    b0 = mean + components[0] * (projected[:, 0].max() + EPSILON)
    radius = float(np.linalg.norm(projected[:, 1:], axis=1).max())
    return a0, b0, radius


def capsule_approximation(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """最小体积外接胶囊：PCA 初值 + SLSQP 约束优化（参考实现的算法）。

    约束与参考实现等价：``max_v dist(v, ab) − r ≤ 0``。参考实现把右端留了
    ``CONSTRAINT_INFLATION_RATIO`` 的余量（``≤ −inflation``）；这里不需要——
    PCA 初值本身就是可行解，可行性判据直接用 ``≤ 0``。

    和参考实现唯一的实质差别是**给 SLSQP 提供解析约束梯度**（``jac``）。参考实现
    依赖数值差分，而约束里含 ``√(≈1e-8)``，差分会被放大到 1e8 量级，在本机的
    scipy 1.8 上直接崩成 ``r≈1e-7`` 的退化解（甚至报告 success=True）——所以这里
    用 :func:`_capsule_gradient` 的次梯度，并自己复验可行性。

    返回 ``(a, b, r, 优化是否给出可行解)``；不可行时退回 PCA 可行初值
    （半径 = 所有顶点到主轴距离的最大值，一定能包住）。
    """
    from scipy import optimize

    vertices = np.asarray(vertices, dtype=float)
    a0, b0, r0 = pca_approximation(vertices)
    x0 = np.concatenate([a0, b0, [r0]])

    def objective(x: np.ndarray) -> float:
        return capsule_volume(x[:3], x[3:6], x[6])

    def constraint(x: np.ndarray) -> float:
        return float(distance_points_segment(vertices, x[:3], x[3:6]).max()) - x[6]

    def constraint_jacobian(x: np.ndarray) -> np.ndarray:
        _, gradient, _ = _capsule_gradient(vertices, x[:3], x[3:6])
        return np.concatenate([gradient, [-1.0]])

    result = optimize.minimize(
        objective, x0, method="SLSQP",
        constraints=[dict(type="ineq", fun=constraint, jac=constraint_jacobian)],
        options=dict(maxiter=config.CAPSULE_MAX_ITER, ftol=config.CAPSULE_FTOL),
    )
    a, b, r = result.x[:3], result.x[3:6], float(result.x[6])
    # SLSQP 有时会「报告成功」但解是退化的（r≈0、顶点其实在外面），必须自己验一遍
    feasible = (bool(np.isfinite(result.x).all()) and r > 0.0
                and constraint(result.x) <= EPSILON)
    if not feasible:
        if not QUIET:
            print(f"    NOTE: SLSQP 未给出可行解（{result.message}），退回 PCA 可行初值")
        return a0, b0, max(float(r0), config.CAPSULE_MIN_RADIUS), False
    return a, b, r, True


def feasible_capsule(vertices: np.ndarray, a: np.ndarray, b: np.ndarray,
                     iterations: int | None = None, tolerance: float = 1e-12
                     ) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """次梯度下降：直接最小化 ``r(a, b) = max_v dist(v, ab)``。

    和 :func:`capsule_approximation`（参考实现的 SLSQP 形式）互补：那条路优化的
    是体积（带一个非凸的端点参数化），这里直接压「最粗处」，两个都跑、取体积小的
    那个。参数化不同、目标一致：最小外接胶囊就是最小体积胶囊。

    每步回溯线搜索、半径单调不增，所以返回的胶囊必定可行（不漏碰）。
    """
    points = np.asarray(vertices, dtype=float)
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    limit = config.CAPSULE_MAX_ITER if iterations is None else iterations
    step = 0.25 * max(float(np.linalg.norm(b - a)), float(np.abs(points).max()), 1e-6)
    radius = float(distance_points_segment(points, a, b).max())

    for _ in range(limit):
        _, gradient, _ = _capsule_gradient(points, a, b)
        magnitude = float(np.linalg.norm(gradient))
        if magnitude <= 1e-15:
            break
        direction = gradient / magnitude
        shrink = 1.0
        moved = False
        for _ in range(48):
            candidate_a = a - step * direction[:3]
            candidate_b = b - step * direction[3:]
            candidate = float(distance_points_segment(points, candidate_a, candidate_b).max())
            if candidate < radius - 1e-15:
                a, b, radius = candidate_a, candidate_b, candidate
                moved = True
                break
            step *= 0.5 * shrink
            shrink *= 0.5 if shrink > 1e-3 else 1.0
        if not moved:
            break
        step *= 1.6                                     # 走通了就放大步长，加快收敛

    return a, b, radius, bool(radius <= float(distance_points_segment(points, a, b).max())
                              + tolerance)


def approximate_mesh(filename: str | Path, lMg: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """读网格 → 顶点变到连杆坐标系 → 拟合胶囊（对应参考实现的同名函数）。

    ``lMg``：连杆坐标系 → 网格坐标系的 4x4 变换（URDF 里 ``<collision><origin>``
    的位姿）。网格本来就建在连杆坐标系里时传单位阵或 None。
    """
    mesh = load_mesh(filename)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if lMg is not None:
        vertices = vertices @ np.asarray(lMg, dtype=float)[:3, :3].T + np.asarray(lMg)[:3, 3]
    a, b, r, _ = capsule_approximation(vertices)
    return a, b, r


def load_mesh(filename: str | Path) -> trimesh.Trimesh:
    """读网格。

    优先用 ``hppfcl.MeshLoader``（参考实现用的就是它，走同一套解析器）；hppfcl
    不在时退回 trimesh。拟合只用顶点，所以这里只要顶点对得上就够了。
    """
    try:
        import hppfcl

        loaded = hppfcl.MeshLoader().load(str(filename), np.ones(3))
        if loaded.num_vertices:
            vertices = np.asarray([loaded.vertices(i) for i in range(loaded.num_vertices)],
                                  dtype=float)
            return trimesh.Trimesh(vertices=vertices, faces=np.zeros((0, 3), dtype=np.int64),
                                   process=False)
    except Exception as exc:                            # hppfcl 缺失/读失败都不致命
        print(f"    NOTE: hppfcl 读网格失败（{exc}），改用 trimesh")
    return trimesh.load(str(filename), force="mesh", process=False)


# --------------------------------------------------------------------------- #
# 从胶囊参数到 URDF / MuJoCo 的位姿
# --------------------------------------------------------------------------- #
def capsule_frame(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """胶囊 → ``(中点, 旋转矩阵)``：旋转矩阵把局部 +Z 转到胶囊轴线方向。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    center = (a + b) / 2.0
    axis = b - a
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-15:
        return center, np.eye(3)
    z = axis / norm
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return center, np.column_stack([x, y, z])


def rpy_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵 → URDF 的 rpy（固定轴 XYZ，等价于 R = Rz·Ry·Rx）"""
    matrix = np.asarray(rotation, dtype=float)
    pitch = np.arctan2(-matrix[2, 0],
                       float(np.hypot(matrix[0, 0], matrix[1, 0])))
    if abs(np.cos(pitch)) < 1e-9:                       # 万向锁
        roll = 0.0
        yaw = np.arctan2(matrix[0, 1], matrix[1, 1])
    else:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.array([roll, pitch, yaw])


def quat_from_rotation(rotation: np.ndarray) -> list[float]:
    """旋转矩阵 → 四元数 (w, x, y, z)，MuJoCo 的 quat 顺序"""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    w, x, y, z = trimesh.transformations.quaternion_from_matrix(transform)
    return [float(w), float(x), float(y), float(z)]


# --------------------------------------------------------------------------- #
# 单个连杆 / 批量
# --------------------------------------------------------------------------- #
def capsule_one(stem: str, margin: float = config.CAPSULE_MARGIN,
                force: bool = False, write_mesh: bool | None = None) -> dict:
    """给一个连杆拟合胶囊，写出 json（+可选 obj），返回参数字典"""
    target = config.capsule_path(stem)
    if target.is_file() and not force:
        print(f"  {stem:10s} 已有 {target.name}，跳过（--force 可重算）")
        return json.loads(target.read_text())

    source = config.visual_mesh(stem)
    started = time.time()
    mesh = load_mesh(source)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        raise ValueError(f"读不出网格顶点：{source}")

    # 拟合只用凸包顶点（最大距离必然出现在凸包顶点上）
    hull = trimesh.convex.convex_hull(vertices)
    hull_vertices = np.asarray(hull.vertices, dtype=float)

    # 两条路都跑，取体积小的那个：
    #   A) 参考实现的 SLSQP（体积目标 + 解析次梯度约束）
    #   B) 次梯度下降（直接压 max 距离，单调可行）
    a_slsqp, b_slsqp, r_slsqp, converged = capsule_approximation(hull_vertices)
    a0, b0, _ = pca_approximation(hull_vertices)
    a_grad, b_grad, r_grad, feasible = feasible_capsule(hull_vertices, a0, b0)
    if capsule_volume(a_grad, b_grad, r_grad) < capsule_volume(a_slsqp, b_slsqp, r_slsqp):
        a, b, fitted_radius, method = a_grad, b_grad, r_grad, "subgradient"
    else:
        a, b, fitted_radius, method = a_slsqp, b_slsqp, r_slsqp, "slsqp"
    r = max(fitted_radius + margin, config.CAPSULE_MIN_RADIUS)

    # 复验用的是**原始**网格（不是简化副本）：简化会让几何缩水零点几毫米，
    # 用原件复验才能保证「胶囊包住零件」这句话对原始 CAD 也成立。
    original = config.visual_path(stem)
    max_surface_distance = float(distance_points_segment(vertices, a, b).max())
    if original.is_file() and original != source:
        original_vertices = np.asarray(load_mesh(original).vertices, dtype=float)
        original_distance = float(distance_points_segment(original_vertices, a, b).max())
        if original_distance > max_surface_distance:
            if not QUIET:
                print(f"    NOTE: {stem} 简化副本比原件略瘦 "
                      f"({(original_distance - max_surface_distance) * 1000:+.3f}mm)，"
                      f"按原件放大半径")
            max_surface_distance = original_distance
            vertices = original_vertices                  # 贴合度也按原件算
    if max_surface_distance > r:                          # 兜底：宁可胖一点也不漏
        if not QUIET:
            print(f"    WARNING: {stem} 复验穿透 {max_surface_distance - r:.3e} m，"
                  f"半径自动放大到刚好包住")
        r = max_surface_distance + margin

    length = float(np.linalg.norm(b - a))
    center, rotation = capsule_frame(a, b)
    volume = capsule_volume(a, b, r)
    # 网格体积按凸包体积算（STL 面片朝向不一定一致，直接用 trimesh 的 volume 会翻车）
    hull_volume = float(abs(hull.volume))

    info = dict(
        stem=stem,
        source=source.name,
        vertices=int(len(vertices)),
        hull_vertices=int(len(hull_vertices)),
        # 胶囊本体
        a=[float(v) for v in a],
        b=[float(v) for v in b],
        radius=r,
        length=length,
        half_length=length / 2.0,
        # 给 URDF/MJCF 用的位姿
        center=[float(v) for v in center],
        rotation=[[float(v) for v in row] for row in rotation],
        rpy=[float(v) for v in rpy_from_rotation(rotation)],
        quat=quat_from_rotation(rotation),
        # 贴合度 / 复验
        fitted_radius=fitted_radius,
        margin=margin,
        max_surface_distance=max_surface_distance,
        clearance=float(r - max_surface_distance),
        capsule_volume=volume,
        hull_volume=hull_volume,
        fill_ratio=float(hull_volume / volume) if volume else float("nan"),
        method=method,
        slsqp_ok=converged,
        subgradient_ok=feasible,
        seconds=time.time() - started,
    )
    if write_mesh is None:
        write_mesh = config.CAPSULE_WRITE_MESH
    if write_mesh:
        mesh_path = config.capsule_mesh_path(stem)
        capsule_mesh(length, r).export(mesh_path)
        info["file"] = mesh_path.name

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"  {stem:10s} {source.name:24s} r={r:.4f} L={length:.4f} "
          f"体积比 {info['fill_ratio']:.3f} 净空 {info['clearance'] * 1000:+.3f}mm "
          f"({info['seconds']:.2f}s)")
    return info


def capsule_mesh(length: float, radius: float, sections: int = 32) -> trimesh.Trimesh:
    """按 ``(length, radius)`` 造胶囊网格，局部 +Z 为轴、原点在中点。

    形状与 URDF 的 ``<cylinder radius length>``（MuJoCo 转成 capsule 图元后）一致，
    ``length`` 是两球心距离，也就是 MuJoCo 的 ``2 * size[1]``。
    """
    if length <= 1e-12:                                 # 退化成一个球
        return trimesh.creation.icosphere(subdivisions=3, radius=radius)
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    caps = [trimesh.creation.icosphere(subdivisions=3, radius=radius)]
    caps.append(caps[0].copy())
    caps[0].apply_translation([0.0, 0.0, length / 2.0])
    caps[1].apply_translation([0.0, 0.0, -length / 2.0])
    return trimesh.util.concatenate([cylinder, *caps])


def capsule_all(only: list[str] | None = None, margin: float = config.CAPSULE_MARGIN,
                force: bool = False, quiet: bool = False) -> dict[str, dict]:
    """批量拟合，返回 ``{stem: 参数}`` 并写出 capsule_params.json"""
    global QUIET
    QUIET = quiet
    config.ensure_dirs()
    stems = discover(only)
    if not stems:
        print(f"WARNING: {config.VISUAL_DIR} 下没有网格可做胶囊近似")
        return {}

    print(f"胶囊近似（Pinocchio capsule-approximation 算法）：{len(stems)} 个网格，"
          f"半径余量 {margin * 1000:.2f}mm")
    report = {}
    for stem in stems:
        report[stem] = capsule_one(stem, margin=margin, force=force)

    if config.CAPSULE_WRITE_PARAMS:
        params_file = config.CAPSULE_DIR / config.CAPSULE_PARAMS_FILE
        params_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"参数已写入 {params_file.relative_to(config.MODEL_DIR)}")

    print(f"完成：{len(report)} 个胶囊，输出到 {config.CAPSULE_DIR}")
    return report


def discover(only: list[str] | None = None) -> list[str]:
    """要拟合的连杆 stem（与凸分解一致：去重排序、跳过简化副本）"""
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
    parser = argparse.ArgumentParser(
        description="用 Pinocchio 的 capsule-approximation 算法给连杆拟合胶囊（椭圆）近似。")
    parser.add_argument("--only", nargs="*", default=None, help="只处理这些连杆")
    parser.add_argument("--margin", type=float, default=config.CAPSULE_MARGIN,
                        help=f"半径余量（米，默认 {config.CAPSULE_MARGIN}）")
    parser.add_argument("--force", action="store_true", help="已有结果也重算")
    parser.add_argument("--quiet", action="store_true",
                        help="不打印 SLSQP 未收敛之类的中间提示")
    args = parser.parse_args()

    capsule_all(only=args.only, margin=args.margin, force=args.force, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
