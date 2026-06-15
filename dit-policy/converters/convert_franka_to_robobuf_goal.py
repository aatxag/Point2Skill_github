#!/usr/bin/env python3
"""
Como ejecutar:
python3 convert_franka_to_robobuf_goal.py \
  --dataset_dir pick_coffee \
  --out_path data_robobuf/pick_coffee_robobuf_goal \
  --calib_dir my_data \
  --state-source q \
  --action-source q_next \
  --gripper-state cmd_width \
  --norm gaussian \
  --goal-norm gaussian \
  --save-raw-goal


Converter para demos Franka FR3 -> robobuf, añadiendo un 3D goal point por timestep.

Entrada esperada:

    demos/pick_coffee/
      episodes/
        episode_0000/
          cam0_256/
          cam0_depth/
          cam1_256/
          traj.npz
          contact_pixel.json

Calibración esperada:

    my_data/
      hand_eye_result.yaml
      camera_intrinsics.yaml    # recomendado, o pasar fx/fy/cx/cy por CLI

Salida:

    OUT_DIR/
      buf.pkl
      ac_norm.json
      goal_norm.json
      conversion_stats.json

Cada timestep queda como:

    obs = {
        "state": q + gripper,
        "enc_cam_0": wrist image JPEG bytes,
        "enc_cam_1": external image JPEG bytes,
        "goal_point": 3D point, shape (3,),
    }

    transition = (obs, action, reward)

Goal point:
    - contact_pixel.json da (u, v) y frame_index donde el gripper ya está cerrado.
    - cam0_depth/frame_index.png da depth.
    - intrínsecos back-proyectan pixel -> p_cam.
    - hand_eye T_cam_to_ee transforma p_cam -> p_ee_at_grasp.
    - ee_pose_at_grasp transforma p_ee_at_grasp -> p_base.
    - para cada t:
        antes del grasp:
            goal_point(t) = inv(T_base_ee(t)) @ p_base
        después del grasp:
            goal_point(t) = p_ee_at_grasp

"""

import argparse
import json
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────

OPEN_WIDTH = 0.08
CLOSE_WIDTH = 0.005
# Depth values >= this threshold are treated as invalid/no-data.
# RealSense depth is usually uint16 in millimeters; 65535 can appear as no-data.
DEPTH_MAX_VALID_MM = 5000
TARGET_HZ = 10.0
TARGET_DT = 1.0 / TARGET_HZ
RESAMPLE_TOLERANCE = 0.55

SYNC_TIMESTAMP_KEYS = [
    "cam1_timestamps",
    "arm_timestamps",
    "ee_timestamps",
]


# ─────────────────────────────────────────────────────────────
# Utilidades imagen
# ─────────────────────────────────────────────────────────────

# Funcion para redimensionar la imagen para ser apropiado para buf
def resize_and_encode(img_path: Path, size: Tuple[int, int] = (256, 256), jpeg_quality: int = 95) -> bytes:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer imagen: {img_path}")

    if img.shape[:2] != (size[1], size[0]):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError(f"Error codificando imagen: {img_path}")

    return encoded.tobytes()


def parse_frame_index_from_path(p: Path) -> int:
    return int(p.stem)


# ─────────────────────────────────────────────────────────────
# Poses y transformaciones
# ─────────────────────────────────────────────────────────────

# De quaternion a matriz
def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n

    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )

# Mis poses son pose7: [x, y, z, qx, qy, qz, qw] funcion para transformar
def pose7_to_T_base_ee(pose: np.ndarray) -> np.ndarray:
    """
    pose = [x, y, z, qx, qy, qz, qw]
    Devuelve T_base_ee: transforma puntos de EE -> base.
    """
    pose = np.asarray(pose, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_R(pose[3:7])
    T[:3, 3] = pose[:3]
    return T

# Aplicar transformacion a u punto
def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    ph = np.ones(4, dtype=np.float64)
    ph[:3] = np.asarray(p, dtype=np.float64)
    return (T @ ph)[:3]


# ─────────────────────────────────────────────────────────────
# Calibración
# ─────────────────────────────────────────────────────────────

def find_file(calib_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = calib_dir / name
        if p.exists():
            return p
    return None


def load_hand_eye(calib_dir: Path, hand_eye_yaml: Optional[str]) -> np.ndarray:
    if hand_eye_yaml:
        path = Path(hand_eye_yaml).expanduser().resolve()
    else:
        path = find_file(calib_dir, ["hand_eye_result.yaml", "hand_eye.yaml", "calibration.yaml"])

    if path is None or not path.exists():
        raise FileNotFoundError(
            "No encontré hand_eye_result.yaml. Usa --calib_dir my_data o --hand_eye_yaml PATH."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "T_cam_to_ee_4x4" not in data:
        raise KeyError(f"{path} no contiene T_cam_to_ee_4x4")

    T = np.asarray(data["T_cam_to_ee_4x4"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam_to_ee_4x4 debe ser 4x4, recibido {T.shape}")

    print(f"Hand-eye loaded: {path}")
    return T


def parse_intrinsics_yaml(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if all(k in data for k in ["fx", "fy", "cx", "cy"]):
        out = {
            "fx": float(data["fx"]),
            "fy": float(data["fy"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
        }
        if "width" in data:
            out["width"] = float(data["width"])
        if "height" in data:
            out["height"] = float(data["height"])
        return out

    if "camera_matrix" in data and "data" in data["camera_matrix"]:
        K = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        out = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }
        if "image_width" in data:
            out["width"] = float(data["image_width"])
        if "image_height" in data:
            out["height"] = float(data["image_height"])
        return out

    # ROS CameraInfo dump:
    # k: [fx,0,cx,0,fy,cy,0,0,1]
    if "k" in data:
        K = np.asarray(data["k"], dtype=np.float64).reshape(3, 3)
        out = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }
        if "width" in data:
            out["width"] = float(data["width"])
        if "height" in data:
            out["height"] = float(data["height"])
        return out

    raise KeyError(f"No pude leer intrínsecos de {path}. Esperaba fx/fy/cx/cy o camera_matrix.data o k.")


def load_intrinsics(calib_dir: Path, img_size: int) -> Dict[str, float]:
    path = calib_dir / "camera_intrinsics.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Guarda primero los intrínsecos en my_data/camera_intrinsics.yaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Acepta el YAML que hemos generado desde CameraInfo:
    # fx, fy, cx, cy, width, height
    if all(k in data for k in ["fx", "fy", "cx", "cy"]):
        intr = {
            "fx": float(data["fx"]),
            "fy": float(data["fy"]),
            "cx": float(data["cx"]),
            "cy": float(data["cy"]),
            "width": float(data.get("width", img_size)),
            "height": float(data.get("height", img_size)),
        }
    elif "camera_matrix" in data and "data" in data["camera_matrix"]:
        K = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        intr = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "width": float(data.get("width", img_size)),
            "height": float(data.get("height", img_size)),
        }
    else:
        raise KeyError(
            f"{path} debe contener fx/fy/cx/cy o camera_matrix.data"
        )

    # Si el CameraInfo era de la imagen original y el recorder guarda 256x256,
    # escalamos K a la resolución guardada.
    w0 = intr["width"]
    h0 = intr["height"]

    if w0 != img_size or h0 != img_size:
        sx = float(img_size) / float(w0)
        sy = float(img_size) / float(h0)

        intr["fx"] *= sx
        intr["cx"] *= sx
        intr["fy"] *= sy
        intr["cy"] *= sy

        intr["scaled_from"] = [float(w0), float(h0)]
        intr["width"] = float(img_size)
        intr["height"] = float(img_size)

    print(
        f"Intrinsics loaded from {path}: "
        f"fx={intr['fx']:.3f}, fy={intr['fy']:.3f}, "
        f"cx={intr['cx']:.3f}, cy={intr['cy']:.3f}, "
        f"width={intr['width']}, height={intr['height']}"
    )

    return intr
# ─────────────────────────────────────────────────────────────
# Gripper, estado, acciones
# ─────────────────────────────────────────────────────────────

def get_gripper_cmd_width(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "gripper_cmd" in data.files:
        cmd = data["gripper_cmd"].astype(np.float32)
        return np.where(cmd > 0, OPEN_WIDTH, CLOSE_WIDTH).astype(np.float32)

    if "gripper" not in data.files:
        raise KeyError("No existe ni gripper_cmd ni gripper en traj.npz")

    g = data["gripper"].astype(np.float32)
    cmd_open = g > np.median(g)
    return np.where(cmd_open, OPEN_WIDTH, CLOSE_WIDTH).astype(np.float32)


def get_gripper_measured(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "gripper" not in data.files:
        return get_gripper_cmd_width(data)
    return data["gripper"].astype(np.float32)


def fill_nan_forward_backward(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    n, d = out.shape

    for j in range(d):
        col = out[:, j]
        valid = np.where(~np.isnan(col))[0]
        if len(valid) == 0:
            out[:, j] = 0.0
            continue

        first = valid[0]
        if first > 0:
            col[:first] = col[first]

        last = col[first]
        for i in range(first + 1, n):
            if np.isnan(col[i]):
                col[i] = last
            else:
                last = col[i]

        out[:, j] = col

    return out

# Construccion del estado

def build_state(
    q: np.ndarray,
    ee_pose: Optional[np.ndarray],
    gripper_for_state: np.ndarray,
    state_source: str,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    g = np.asarray(gripper_for_state, dtype=np.float32).reshape(-1, 1)

    if state_source == "q":
        state = np.concatenate([q, g], axis=1)
    elif state_source == "ee_pose":
        if ee_pose is None:
            raise KeyError("state_source=ee_pose pero traj.npz no contiene ee_pose")
        state = np.concatenate([ee_pose.astype(np.float32), g], axis=1)
    elif state_source == "q_ee":
        if ee_pose is None:
            raise KeyError("state_source=q_ee pero traj.npz no contiene ee_pose")
        state = np.concatenate([q, ee_pose.astype(np.float32), g], axis=1)
    else:
        raise ValueError(f"state_source no soportado: {state_source}")

    if np.isnan(state).any():
        state = fill_nan_forward_backward(state)

    return state.astype(np.float32)


def build_absolute_actions(
    q: np.ndarray,
    ee_pose: Optional[np.ndarray],
    gripper_cmd_width: np.ndarray,
    action_source: str,
) -> np.ndarray:
    g_next = np.asarray(gripper_cmd_width[1:], dtype=np.float32).reshape(-1, 1)

    if action_source == "q_next":
        actions = np.concatenate([q[1:].astype(np.float32), g_next], axis=1)
    elif action_source == "ee_pose_next":
        if ee_pose is None:
            raise KeyError("action_source=ee_pose_next pero traj.npz no contiene ee_pose")
        ee = fill_nan_forward_backward(np.asarray(ee_pose, dtype=np.float32))
        actions = np.concatenate([ee[1:], g_next], axis=1)
    elif action_source == "q_ee_next":
        if ee_pose is None:
            raise KeyError("action_source=q_ee_next pero traj.npz no contiene ee_pose")
        ee = fill_nan_forward_backward(np.asarray(ee_pose, dtype=np.float32))
        actions = np.concatenate([q[1:].astype(np.float32), ee[1:], g_next], axis=1)
    else:
        raise ValueError(f"action_source no soportado: {action_source}")

    return actions.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Goal point
# ─────────────────────────────────────────────────────────────

# Se lee el depth del pixel correspondiente
# se asume que esta en milimetros, por lo tanto se pasa a metros

def find_nearest_valid_depth_pixel(
    depth_img: np.ndarray,
    u: int,
    v: int,
    radius: int,
):
    """
    Busca el píxel de depth válido más cercano a (u, v).

    Criterio:
      1. depth > 0
      2. depth < DEPTH_MAX_VALID_MM
      3. mínima distancia euclídea al pixel pedido
      4. si hay empate, menor depth

    Devuelve:
      (best_u, best_v, best_depth_raw) o None.
    """
    h, w = depth_img.shape[:2]

    if not (0 <= u < w and 0 <= v < h):
        return None

    best = None
    best_dist2 = None
    best_depth = None

    u0 = max(0, int(u) - int(radius))
    u1 = min(w, int(u) + int(radius) + 1)
    v0 = max(0, int(v) - int(radius))
    v1 = min(h, int(v) + int(radius) + 1)

    for vv in range(v0, v1):
        for uu in range(u0, u1):
            d = float(depth_img[vv, uu])

            if not np.isfinite(d):
                continue
            if d <= 0 or d >= DEPTH_MAX_VALID_MM:
                continue

            dist2 = (uu - u) ** 2 + (vv - v) ** 2

            if best is None:
                best = (uu, vv, d)
                best_dist2 = dist2
                best_depth = d
                continue

            if dist2 < best_dist2:
                best = (uu, vv, d)
                best_dist2 = dist2
                best_depth = d
            elif dist2 == best_dist2 and d < best_depth:
                best = (uu, vv, d)
                best_dist2 = dist2
                best_depth = d

    return best

def read_depth_meters(
    depth_path: Path,
    u: int,
    v: int,
    depth_scale: float,
    depth_window: int,
) -> float:
    """
    Lee el depth en (u, v). Si ese pixel no es válido, busca el depth válido
    más cercano dentro de un radio depth_window.

    Esto sustituye la mediana local por la búsqueda robusta del vecino válido
    más cercano, para no mezclar profundidad de fondo/objeto cuando el pixel
    exacto cae en un agujero de depth.
    """
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"No se pudo leer depth: {depth_path}")

    if depth.ndim != 2:
        raise ValueError(f"Depth image debe ser monocanal. Shape recibido: {depth.shape}")

    h, w = depth.shape[:2]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))

    d0 = float(depth[v, u])
    if np.isfinite(d0) and d0 > 0 and d0 < DEPTH_MAX_VALID_MM:
        return float(d0 * depth_scale)

    nearest = find_nearest_valid_depth_pixel(
        depth_img=depth,
        u=u,
        v=v,
        radius=max(0, int(depth_window)),
    )

    if nearest is None:
        raise ValueError(
            f"Depth inválido en ({u},{v}) y no hay vecinos válidos "
            f"en radio {depth_window} en {depth_path}"
        )

    used_u, used_v, used_depth = nearest
    return float(used_depth * depth_scale)

# Pas de 2D + depth a un punto 3D en camara con intrinsecas
def backproject_pixel(u: int, v: int, z: float, intr: Dict[str, float]) -> np.ndarray:
    x = (float(u) - intr["cx"]) * float(z) / intr["fx"]
    y = (float(v) - intr["cy"]) * float(z) / intr["fy"]
    return np.array([x, y, float(z)], dtype=np.float64)

# Se lee el pixel anotado
def load_contact_pixel(ep: Path) -> Dict:
    path = ep / "contact_pixel.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta annotate_contact_pixels.py primero.")

    with open(path, "r", encoding="utf-8") as f:
        c = json.load(f)

    for k in ["u", "v", "frame_index"]:
        if k not in c:
            raise KeyError(f"{path} no contiene '{k}'")

    if c.get("camera", "cam0") != "cam0":
        raise ValueError(f"{path}: por ahora el goal 3D usa cam0/cam0_depth, recibido camera={c.get('camera')}")

    return c


def compute_episode_goal_context(
    ep: Path,
    ee_pose_raw: np.ndarray,
    T_cam_to_ee: np.ndarray,
    intr: Dict[str, float],
    depth_scale: float,
    depth_window: int,
) -> Dict:
    """
    Calcula:
      - contact_frame_idx
      - p_contact_cam
      - p_contact_ee_grasp
      - p_contact_base
    """
    contact = load_contact_pixel(ep)

    frame_idx = int(contact["frame_index"])
    u = int(contact["u"])
    v = int(contact["v"])

    if frame_idx < 0 or frame_idx >= len(ee_pose_raw):
        raise IndexError(f"{ep.name}: frame_index={frame_idx} fuera de ee_pose len={len(ee_pose_raw)}")

    depth_path = ep / "cam0_depth" / f"{frame_idx:06d}.png"
    z = read_depth_meters(depth_path, u, v, depth_scale=depth_scale, depth_window=depth_window)

    p_cam = backproject_pixel(u, v, z, intr)

    # Una vez se ha hecho backprojection a punto 3D en camara ahora pasar a base
    ee_pose_raw = fill_nan_forward_backward(np.asarray(ee_pose_raw, dtype=np.float32)).astype(np.float64)
    T_base_ee_grasp = pose7_to_T_base_ee(ee_pose_raw[frame_idx])

    # Con T de Hand Eye pasamos de cam a ee
    p_ee_grasp = transform_point(T_cam_to_ee, p_cam)
    # Con T de FK pasamos de ee a base
    p_base = transform_point(T_base_ee_grasp, p_ee_grasp)

    return {
        "contact": contact,
        "contact_frame_idx": int(frame_idx),
        "depth_m": float(z),
        "p_contact_cam": p_cam.astype(np.float64),
        "p_contact_ee_grasp": p_ee_grasp.astype(np.float64),
        "p_contact_base": p_base.astype(np.float64),
    }


# SE LLAMA PARA CADA TIMESTEP
def compute_goal_points_for_segment(
    ee_pose_segment: np.ndarray,
    cam0_files_segment: List[Path],
    goal_ctx: Dict,
) -> np.ndarray:
    """
    Devuelve goal_point por cada estado del segmento, shape (N, 3).

    Se compara el índice raw de la imagen con contact_frame_idx:
      raw_idx < contact_frame_idx -> antes del grasp
      raw_idx >= contact_frame_idx -> después del grasp
    """
    if ee_pose_segment is None:
        raise KeyError("Se necesita ee_pose para calcular goal_point")

    ee_pose_segment = fill_nan_forward_backward(np.asarray(ee_pose_segment, dtype=np.float32)).astype(np.float64)

    contact_frame_idx = int(goal_ctx["contact_frame_idx"])
    p_base = goal_ctx["p_contact_base"]
    p_ee_grasp = goal_ctx["p_contact_ee_grasp"]

    goals = []
    for pose, img_path in zip(ee_pose_segment, cam0_files_segment):
        raw_idx = parse_frame_index_from_path(img_path)

        if raw_idx < contact_frame_idx:
            T_base_ee_t = pose7_to_T_base_ee(pose)
            p_ee_t = transform_point(np.linalg.inv(T_base_ee_t), p_base)
        else:
            p_ee_t = p_ee_grasp

        goals.append(p_ee_t.astype(np.float32))

    return np.asarray(goals, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Timestamps / resampling
# ─────────────────────────────────────────────────────────────

def check_timestamps(data: np.lib.npyio.NpzFile, ep_name: str, warn_ms: float = 200.0) -> bool:
    if "cam0_timestamps" not in data.files:
        print(f"[{ep_name}] ERROR: sin cam0_timestamps")
        return False

    ref = data["cam0_timestamps"].astype(np.float64)

    for key in SYNC_TIMESTAMP_KEYS:
        if key not in data.files:
            continue

        ts = data[key].astype(np.float64)
        n = min(len(ref), len(ts))
        if n == 0:
            continue

        diff_ms = np.abs(ts[:n] - ref[:n]) * 1000.0
        print(
            f"[{ep_name}] {key:<18} "
            f"mean={diff_ms.mean():6.1f}ms max={diff_ms.max():6.1f}ms"
        )

        if diff_ms.max() > warn_ms:
            print(f"[{ep_name}] ADVERTENCIA: {key} tiene desfase alto (>{warn_ms:.0f}ms)")

    return True


def interp_array(t_valid: np.ndarray, ts: np.ndarray, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return np.column_stack([
        np.interp(t_valid, ts, arr[:, j]) for j in range(arr.shape[1])
    ]).astype(np.float32)

# Guardamos la informacion para 10Hz, tomando los puntos mas cercanos a cada 100ms, y descartando los que no tengan un punto cercano suficiente.
def resample_episode(
    cam0_ts: np.ndarray,
    q: np.ndarray,
    ee_pose: Optional[np.ndarray],
    gripper_measured: np.ndarray,
    gripper_cmd_width: np.ndarray,
    cam0_files: List[Path],
    cam1_files: List[Path],
    target_dt: float = TARGET_DT,
    tolerance: float = RESAMPLE_TOLERANCE,
):
    ts = np.asarray(cam0_ts, dtype=np.float64)
    t0, t_end = float(ts[0]), float(ts[-1])
    t_grid = np.arange(t0, t_end, target_dt)

    valid_grid_times = []
    nearest_idx = []

    for t in t_grid:
        diffs = np.abs(ts - t)
        idx = int(np.argmin(diffs))
        if diffs[idx] < tolerance * target_dt:
            valid_grid_times.append(t)
            nearest_idx.append(idx)

    n_dropped = len(t_grid) - len(valid_grid_times)

    if len(nearest_idx) < 2:
        return [], n_dropped

    t_valid = np.asarray(valid_grid_times, dtype=np.float64)
    nearest_idx = np.asarray(nearest_idx, dtype=np.int64)

    q_out = interp_array(t_valid, ts, q)
    ee_out = interp_array(t_valid, ts, ee_pose) if ee_pose is not None else None
    g_meas_out = np.interp(t_valid, ts, gripper_measured).astype(np.float32)

    g_cmd_out = gripper_cmd_width[nearest_idx].astype(np.float32)
    cam0_out = [cam0_files[i] for i in nearest_idx]
    cam1_out = [cam1_files[i] for i in nearest_idx]

    split_at = np.where(np.diff(t_valid) > 1.5 * target_dt)[0] + 1
    boundaries = [0] + split_at.tolist() + [len(t_valid)]

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue

        segments.append(
            {
                "q": q_out[start:end],
                "ee_pose": ee_out[start:end] if ee_out is not None else None,
                "gripper_measured": g_meas_out[start:end],
                "gripper_cmd_width": g_cmd_out[start:end],
                "cam0_files": cam0_out[start:end],
                "cam1_files": cam1_out[start:end],
                "t_start": float(t_valid[start]),
                "t_end": float(t_valid[end - 1]),
            }
        )

    return segments, n_dropped


def make_segments_without_resample(
    q: np.ndarray,
    ee_pose: Optional[np.ndarray],
    gripper_measured: np.ndarray,
    gripper_cmd_width: np.ndarray,
    cam0_files: List[Path],
    cam1_files: List[Path],
):
    return [
        {
            "q": q,
            "ee_pose": ee_pose,
            "gripper_measured": gripper_measured,
            "gripper_cmd_width": gripper_cmd_width,
            "cam0_files": cam0_files,
            "cam1_files": cam1_files,
            "t_start": None,
            "t_end": None,
        }
    ]


# ─────────────────────────────────────────────────────────────
# Normalización
# ─────────────────────────────────────────────────────────────

# Como en los convertidores de ejemplo, se normalizan in-place las referencias.
def gaussian_norm_inplace(arr_refs: List[np.ndarray]) -> Dict[str, list]:
    arr = np.asarray(arr_refs, dtype=np.float32)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std[std == 0] = 1e-17

    for a in arr_refs:
        a -= mean
        a /= std

    return {"loc": mean.tolist(), "scale": std.tolist(), "type": "gaussian"}


def max_min_norm_inplace(arr_refs: List[np.ndarray]) -> Dict[str, list]:
    arr = np.asarray(arr_refs, dtype=np.float32)
    max_v = np.max(arr, axis=0)
    min_v = np.min(arr, axis=0)

    mid = (max_v + min_v) / 2.0
    delta = (max_v - min_v) / 2.0
    delta[delta == 0] = 1e-17

    for a in arr_refs:
        a -= mid
        a /= delta

    return {"loc": mid.tolist(), "scale": delta.tolist(), "type": "max_min"}


def no_norm(arr_refs: List[np.ndarray]) -> Dict[str, list]:
    return {
        "loc": np.zeros_like(arr_refs[0]).tolist(),
        "scale": np.ones_like(arr_refs[0]).tolist(),
        "type": "none",
    }


def normalize_refs(arr_refs: List[np.ndarray], norm: str, label: str) -> Dict[str, list]:
    if norm == "gaussian":
        print(f"Using gaussian norm for {label}")
        return gaussian_norm_inplace(arr_refs)
    if norm == "max_min":
        print(f"Using max_min norm for {label}")
        return max_min_norm_inplace(arr_refs)
    if norm == "none":
        print(f"No normalization for {label}")
        return no_norm(arr_refs)
    raise ValueError(f"norm no soportado: {norm}")


# ─────────────────────────────────────────────────────────────
# Conversión
# ─────────────────────────────────────────────────────────────

def load_task_instruction(episode_dir: Path, default_instruction: str) -> Optional[str]:
    for name in ("instruction.txt", "language.txt", "task.txt"):
        p = episode_dir / name
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            if text:
                return text

    return default_instruction if default_instruction else None


# ─────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL DE CONVERSIÓN
# ──────────────────────────────────────────────────────────

"""
1. Comprueba que existen:
   - traj.npz
   - cam0_256
   - cam1_256
   - cam0_depth
   - contact_pixel.json

2. Carga:
   - q
   - ee_pose
   - gripper
   - gripper_cmd
   - imágenes

3. Calcula el goal context del episodio:
   - pixel + depth
   - p_cam
   - p_ee_grasp
   - p_base

4. Hace resampling o usa frames raw.

5. Para cada segmento:
   - construye state
   - construye action
   - construye goal_point
   - codifica imágenes
   - guarda transición

6. Al final:
   - normaliza acciones
   - normaliza goal_point
   - guarda buf.pkl
   - guarda ac_norm.json
   - guarda goal_norm.json
   - guarda conversion_stats.json

"""

def convert_dataset(
    dataset_dir: Path,
    out_path: Path,
    calib_dir: Path,
    hand_eye_yaml: Optional[str],
    intrinsics_yaml: Optional[str],
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    intrinsics_width: Optional[int],
    intrinsics_height: Optional[int],
    img_size: int,
    jpeg_quality: int,
    state_source: str,
    action_source: str,
    gripper_state: str,
    norm: str,
    goal_norm: str,
    no_resample: bool,
    task_instruction: str,
    min_frames: int,
    save_reward_last: bool,
    depth_scale: float,
    depth_window: int,
    save_raw_goal: bool,
):
    episodes_dir = dataset_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {episodes_dir}")

    episode_dirs = sorted([p for p in episodes_dir.glob("episode_*") if p.is_dir()])
    if not episode_dirs:
        raise FileNotFoundError(f"No encontré episode_* dentro de: {episodes_dir}")

    T_cam_to_ee = load_hand_eye(calib_dir, hand_eye_yaml)
    intr = load_intrinsics(calib_dir=calib_dir, img_size=img_size)

    out_trajs = []
    all_acs = []
    all_goal_refs = []
    stats = []

    skipped = 0
    total_dropped_frames = 0
    size = (img_size, img_size)

    for ep in tqdm(episode_dirs, desc="Episodes"):
        traj_path = ep / "traj.npz"
        cam0_dir = ep / "cam0_256"
        cam1_dir = ep / "cam1_256"
        depth_dir = ep / "cam0_depth"
        contact_path = ep / "contact_pixel.json"

        if not traj_path.exists():
            print(f"[SKIP] {ep.name}: no existe traj.npz")
            skipped += 1
            continue

        if not cam0_dir.exists() or not cam1_dir.exists():
            print(f"[SKIP] {ep.name}: faltan cam0_256 o cam1_256")
            skipped += 1
            continue

        if not depth_dir.exists():
            print(f"[SKIP] {ep.name}: falta cam0_depth")
            skipped += 1
            continue

        if not contact_path.exists():
            print(f"[SKIP] {ep.name}: falta contact_pixel.json")
            skipped += 1
            continue

        data = np.load(traj_path, allow_pickle=True)

        if "q" not in data.files:
            print(f"[SKIP] {ep.name}: traj.npz no contiene q")
            skipped += 1
            continue

        if "ee_pose" not in data.files:
            print(f"[SKIP] {ep.name}: traj.npz no contiene ee_pose, necesario para goal_point")
            skipped += 1
            continue

        if not no_resample and not check_timestamps(data, ep.name):
            skipped += 1
            continue

        q = data["q"].astype(np.float32)
        ee_pose = data["ee_pose"].astype(np.float32)
        gripper_measured = get_gripper_measured(data)
        gripper_cmd_width = get_gripper_cmd_width(data)

        cam0_files = sorted(cam0_dir.glob("*.jpg"))
        cam1_files = sorted(cam1_dir.glob("*.jpg"))

        lengths = [
            len(q),
            len(ee_pose),
            len(gripper_measured),
            len(gripper_cmd_width),
            len(cam0_files),
            len(cam1_files),
        ]

        t = min(lengths)

        if t < min_frames:
            print(f"[SKIP] {ep.name}: solo {t} frames (< {min_frames})")
            skipped += 1
            continue

        q = q[:t]
        ee_pose = ee_pose[:t]
        gripper_measured = gripper_measured[:t]
        gripper_cmd_width = gripper_cmd_width[:t]
        cam0_files = cam0_files[:t]
        cam1_files = cam1_files[:t]

        try:
            goal_ctx = compute_episode_goal_context(
                ep=ep,
                ee_pose_raw=ee_pose,
                T_cam_to_ee=T_cam_to_ee,
                intr=intr,
                depth_scale=depth_scale,
                depth_window=depth_window,
            )
        except Exception as e:
            print(f"[SKIP] {ep.name}: no se pudo calcular goal context: {e}")
            skipped += 1
            continue

        if no_resample:
            segments = make_segments_without_resample(
                q, ee_pose, gripper_measured, gripper_cmd_width, cam0_files, cam1_files
            )
            n_dropped = 0
        else:
            if "cam0_timestamps" not in data.files or len(data["cam0_timestamps"]) < t:
                print(f"[SKIP] {ep.name}: cam0_timestamps ausentes o demasiado cortos")
                skipped += 1
                continue

            segments, n_dropped = resample_episode(
                cam0_ts=data["cam0_timestamps"][:t],
                q=q,
                ee_pose=ee_pose,
                gripper_measured=gripper_measured,
                gripper_cmd_width=gripper_cmd_width,
                cam0_files=cam0_files,
                cam1_files=cam1_files,
            )
            total_dropped_frames += n_dropped

            if not segments:
                print(f"[SKIP] {ep.name}: ningún segmento válido tras resampling")
                skipped += 1
                continue

        ep_instruction = load_task_instruction(ep, task_instruction)
        ep_valid_segments = 0

        for seg_idx, seg in enumerate(segments):
            q_s = seg["q"]
            ee_s = seg["ee_pose"]
            g_meas_s = seg["gripper_measured"]
            g_cmd_s = seg["gripper_cmd_width"]
            cam0_s = seg["cam0_files"]
            cam1_s = seg["cam1_files"]

            if len(q_s) < min_frames:
                continue

            if gripper_state == "cmd_width":
                gripper_for_state = g_cmd_s
            elif gripper_state == "measured":
                gripper_for_state = g_meas_s
            else:
                raise ValueError(f"gripper_state no soportado: {gripper_state}")

            state = build_state(
                q=q_s,
                ee_pose=ee_s,
                gripper_for_state=gripper_for_state,
                state_source=state_source,
            )

            actions = build_absolute_actions(
                q=q_s,
                ee_pose=ee_s,
                gripper_cmd_width=g_cmd_s,
                action_source=action_source,
            )

            goal_points = compute_goal_points_for_segment(
                ee_pose_segment=ee_s,
                cam0_files_segment=cam0_s,
                goal_ctx=goal_ctx,
            )

            T = len(actions)  # N-1 transiciones
            if T < 1:
                continue

            proc_traj = []
            for i in range(T):
                goal_ref = goal_points[i].astype(np.float32)

                obs = {
                    "state": state[i].astype(np.float32),
                    "enc_cam_0": resize_and_encode(cam0_s[i], size=size, jpeg_quality=jpeg_quality),
                    "enc_cam_1": resize_and_encode(cam1_s[i], size=size, jpeg_quality=jpeg_quality),
                    "goal_point": goal_ref,
                }

                if save_raw_goal:
                    obs["goal_point_raw"] = goal_points[i].astype(np.float32).copy()

                if ep_instruction is not None:
                    obs["language"] = ep_instruction

                obs["episode_name"] = ep.name
                obs["segment_idx"] = int(seg_idx)
                obs["timestep"] = int(i)

                reward = 1.0 if (save_reward_last and i == T - 1) else 0.0

                proc_traj.append((obs, actions[i], float(reward)))
                all_acs.append(actions[i])
                all_goal_refs.append(goal_ref)

            if proc_traj:
                out_trajs.append(proc_traj)
                ep_valid_segments += 1

                raw_goal_norms = np.linalg.norm(goal_points[:T], axis=1)

                stats.append(
                    {
                        "episode": ep.name,
                        "segment_idx": int(seg_idx),
                        "length": int(len(proc_traj)),
                        "raw_frames_in_segment": int(len(q_s)),
                        "state_dim": int(proc_traj[0][0]["state"].shape[0]),
                        "action_dim": int(proc_traj[0][1].shape[0]),
                        "goal_dim": int(proc_traj[0][0]["goal_point"].shape[0]),
                        "contact_frame_idx": int(goal_ctx["contact_frame_idx"]),
                        "contact_u": int(goal_ctx["contact"]["u"]),
                        "contact_v": int(goal_ctx["contact"]["v"]),
                        "contact_depth_m": float(goal_ctx["depth_m"]),
                        "goal_norm_min_raw_m": float(np.min(raw_goal_norms)),
                        "goal_norm_max_raw_m": float(np.max(raw_goal_norms)),
                        "t_start": seg.get("t_start"),
                        "t_end": seg.get("t_end"),
                    }
                )

        if ep_valid_segments == 0:
            skipped += 1

    if not out_trajs:
        raise RuntimeError("No se convirtió ningún episodio/segmento válido.")

    ac_norm = normalize_refs(all_acs, norm=norm, label="actions")
    goal_norm_dict = normalize_refs(all_goal_refs, norm=goal_norm, label="goal_point")

    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "buf.pkl", "wb") as f:
        pkl.dump(out_trajs, f)

    with open(out_path / "ac_norm.json", "w", encoding="utf-8") as f:
        json.dump(ac_norm, f, indent=2)

    with open(out_path / "goal_norm.json", "w", encoding="utf-8") as f:
        json.dump(goal_norm_dict, f, indent=2)

    with open(out_path / "contact_norm.json", "w", encoding="utf-8") as f:
        json.dump(goal_norm_dict, f, indent=2)

# Aqui se guarda informacion para el debug, por si acaso
    conversion_stats = {
        "dataset_dir": str(dataset_dir),
        "out_path": str(out_path),
        "calib_dir": str(calib_dir),
        "num_output_trajectories": int(len(out_trajs)),
        "num_steps": int(len(all_acs)),
        "skipped_episodes_or_segments": int(skipped),
        "dropped_frames_resampling": int(total_dropped_frames),
        "img_size": int(img_size),
        "jpeg_quality": int(jpeg_quality),
        "state_source": state_source,
        "action_source": action_source,
        "gripper_state": gripper_state,
        "action_norm": norm,
        "goal_norm": goal_norm,
        "no_resample": bool(no_resample),
        "target_hz": TARGET_HZ,
        "target_dt": TARGET_DT,
        "resample_tolerance": RESAMPLE_TOLERANCE,
        "save_reward_last": bool(save_reward_last),
        "depth_scale": float(depth_scale),
        "depth_window": int(depth_window),
        "intrinsics_used": intr,
        "T_cam_to_ee_4x4": T_cam_to_ee.tolist(),
        "segments": stats,
    }

    with open(out_path / "conversion_stats.json", "w", encoding="utf-8") as f:
        json.dump(conversion_stats, f, indent=2)

    print()
    print("DONE")
    print(f"output dir:        {out_path}")
    print(f"trajectories:      {len(out_trajs)}")
    print(f"steps:             {len(all_acs)}")
    print(f"skipped:           {skipped}")
    print(f"dropped resample:  {total_dropped_frames}")
    print(f"state_source:      {state_source}")
    print(f"action_source:     {action_source}")
    print(f"state_dim:         {stats[0]['state_dim']}")
    print(f"action_dim:        {stats[0]['action_dim']}")
    print(f"goal_dim:          {stats[0]['goal_dim']}")
    print(f"action_norm:       {norm}")
    print(f"goal_norm:         {goal_norm}")
    print(f"saved:             {out_path / 'buf.pkl'}")
    print(f"saved:             {out_path / 'ac_norm.json'}")
    print(f"saved:             {out_path / 'goal_norm.json'}")
    print(f"saved:             {out_path / 'contact_norm.json'}")
    print(f"saved:             {out_path / 'conversion_stats.json'}")


def inspect_dataset(dataset_dir: Path):
    episodes_dir = dataset_dir / "episodes"
    episode_dirs = sorted([p for p in episodes_dir.glob("episode_*") if p.is_dir()])

    print(f"dataset_dir: {dataset_dir}")
    print(f"episodes_dir: {episodes_dir}")
    print(f"num episodes: {len(episode_dirs)}")

    for ep in episode_dirs[:10]:
        traj_path = ep / "traj.npz"
        cam0_dir = ep / "cam0_256"
        cam1_dir = ep / "cam1_256"
        depth_dir = ep / "cam0_depth"
        contact_path = ep / "contact_pixel.json"

        print()
        print(ep.name)
        print(f"  traj.npz:            {traj_path.exists()}")
        print(f"  contact_pixel.json:  {contact_path.exists()}")
        print(f"  cam0 jpg:            {len(list(cam0_dir.glob('*.jpg'))) if cam0_dir.exists() else 0}")
        print(f"  cam1 jpg:            {len(list(cam1_dir.glob('*.jpg'))) if cam1_dir.exists() else 0}")
        print(f"  cam0 depth png:      {len(list(depth_dir.glob('*.png'))) if depth_dir.exists() else 0}")

        if contact_path.exists():
            with open(contact_path, "r", encoding="utf-8") as f:
                c = json.load(f)
            print(f"  contact: frame={c.get('frame_index')} u={c.get('u')} v={c.get('v')} camera={c.get('camera')}")

        if traj_path.exists():
            data = np.load(traj_path, allow_pickle=True)
            print(f"  keys: {sorted(data.files)}")
            for key in ["q", "ee_pose", "gripper", "gripper_cmd", "cam0_timestamps"]:
                if key in data.files:
                    arr = data[key]
                    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", required=True, help="Ej: pick_coffee")
    parser.add_argument("--out_path", required=True, help="Ej: pick_coffee_robobuf_goal")

    parser.add_argument("--calib_dir", default="my_data", help="Carpeta con hand_eye_result.yaml e intrínsecos.")
    parser.add_argument("--hand_eye_yaml", default=None, help="Ruta directa a hand_eye_result.yaml.")
    parser.add_argument("--intrinsics_yaml", default=None, help="Ruta a camera_intrinsics.yaml/camera_info.yaml.")

    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--intrinsics-width", type=int, default=None, help="Resolución original de los intrínsecos.")
    parser.add_argument("--intrinsics-height", type=int, default=None, help="Resolución original de los intrínsecos.")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--jpeg_quality", type=int, default=95)

    parser.add_argument("--depth-scale", type=float, default=0.001, help="RealSense uint16 mm -> m usa 0.001.")
    parser.add_argument("--depth-window", type=int, default=5, help="Ventana impar para mediana de depth alrededor del pixel.")

    parser.add_argument(
        "--state-source",
        choices=["q", "ee_pose", "q_ee"],
        default="q",
        help="Estado usado en obs['state']. Recomendado: q.",
    )
    parser.add_argument(
        "--action-source",
        choices=["q_next", "ee_pose_next", "q_ee_next"],
        default="q_next",
        help="Acción absoluta objetivo. Recomendado: q_next.",
    )
    parser.add_argument(
        "--gripper-state",
        choices=["cmd_width", "measured"],
        default="cmd_width",
        help="Qué gripper meter en obs['state']. Recomendado: cmd_width.",
    )
    parser.add_argument(
        "--norm",
        choices=["gaussian", "max_min", "none"],
        default="gaussian",
        help="Normalización de acciones.",
    )
    parser.add_argument(
        "--goal-norm",
        choices=["gaussian", "max_min", "none"],
        default="gaussian",
        help="Normalización del obs['goal_point']. Para adaLN se recomienda gaussian.",
    )

    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument("--task-instruction", default="")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--save-reward-last", action="store_true")
    parser.add_argument("--save-raw-goal", action="store_true", help="Guarda también obs['goal_point_raw'] sin normalizar.")
    parser.add_argument("--inspect-only", action="store_true")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    calib_dir = Path(args.calib_dir).expanduser().resolve()

    if args.inspect_only:
        inspect_dataset(dataset_dir)
        return

    convert_dataset(
        dataset_dir=dataset_dir,
        out_path=out_path,
        calib_dir=calib_dir,
        hand_eye_yaml=args.hand_eye_yaml,
        intrinsics_yaml=args.intrinsics_yaml,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        intrinsics_width=args.intrinsics_width,
        intrinsics_height=args.intrinsics_height,
        img_size=args.img_size,
        jpeg_quality=args.jpeg_quality,
        state_source=args.state_source,
        action_source=args.action_source,
        gripper_state=args.gripper_state,
        norm=args.norm,
        goal_norm=args.goal_norm,
        no_resample=args.no_resample,
        task_instruction=args.task_instruction,
        min_frames=args.min_frames,
        save_reward_last=args.save_reward_last,
        depth_scale=args.depth_scale,
        depth_window=args.depth_window,
        save_raw_goal=args.save_raw_goal,
    )


if __name__ == "__main__":
    main()
