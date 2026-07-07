#!/usr/bin/env python3
"""
Conversión recorder sync (2 cámaras + depth) → robobuf con ABSOLUTE ACTIONS + PLACE ANCHOR
Compatible con recorder_node_two_cam_place.py.

Equivalente a convert_to_robobuf_contact_hindsight.py pero para tareas de PLACE:
  - El gripper empieza CERRADO (objeto en mano).
  - El frame de referencia es cuando el gripper ABRE físicamente (release).
  - El place anchor 3D captura la posición del target en frame base.
  - Se retropropaga (world-fixed) a todos los frames del episodio.

VERSIÓN NO-SPLIT: cada episode_XXXX se convierte en una única trayectoria.
No se hace resampling ni se divide por gaps temporales (igual que la versión
de contact). Añade además:
  - --trim_start_motion: recorta la pausa inicial antes del primer movimiento real
  - --dedup_static: elimina frames consecutivos casi duplicados
  - profundidad inválida sin vecino válido → error (episodio se descarta), en
    vez de rellenar con un valor inventado
"""

import argparse
import json
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ── Constantes ──────────────────────────────────────────────────────────────

OPEN_WIDTH  = 0.08
CLOSE_WIDTH = 0.0

SYNC_TIMESTAMP_KEYS = [
    "cam1_timestamps",
    "arm_timestamps",
]

# Pixel de place anchor en imagen 256×256 (u=columna, v=fila)
# Ajustar según visualización del wrist cam durante place
CONTACT_U = 160
CONTACT_V = 110

# Radio de búsqueda si depth(u,v) == 0
DEPTH_SEARCH_RADIUS = 15

DEPTH_MAX_VALID_MM = 5000

# Para place el target puede estar más lejos que en pick.
# Ajustar según los datos reales (usar viz_contact_trajectory sobre un episodio).
CONTACT_DEPTH_MAX_MM = 800

# Intrínseca D435i @ 1280×720, reescalada a 256×256
D435I_FX = 181.685
D435I_FY = 322.924
D435I_CX = 129.035
D435I_CY = 131.825


# ── Utilidades geométricas ─────────────────────────────────────────────────

def load_T_cam_to_ee(yaml_path: Path) -> np.ndarray:
    if not yaml_path.exists():
        raise FileNotFoundError(f"No existe hand-eye YAML: {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if "T_cam_to_ee_4x4" not in data:
        raise KeyError(f"El YAML {yaml_path} no contiene 'T_cam_to_ee_4x4'")
    T = np.array(data["T_cam_to_ee_4x4"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam_to_ee_4x4 debe ser (4,4), recibido {T.shape}")
    return T


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64)
    x, y, z, qx, qy, qz, qw = pose7
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def sanitize_ee_pose(ee_pose_arr: np.ndarray, ep_name: str) -> np.ndarray:
    arr = ee_pose_arr.copy().astype(np.float64)
    norms = np.linalg.norm(arr[:, 3:], axis=1)
    valid = norms > 1e-6
    n_bad = int((~valid).sum())
    if n_bad == 0:
        return arr
    if not valid.any():
        raise RuntimeError(f"[{ep_name}] ee_pose sin ninguna entrada válida")
    print(f"  [{ep_name}] ee_pose: {n_bad} entradas con cuaternión nulo → relleno con vecino válido")
    valid_idx = np.where(valid)[0]
    for i in np.where(~valid)[0]:
        nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
        arr[i] = arr[nearest]
    return arr


def backproject(u: int, v: int, depth_m: float,
                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (float(u) - float(cx)) * float(depth_m) / float(fx)
    y = (float(v) - float(cy)) * float(depth_m) / float(fy)
    z = float(depth_m)
    return np.array([x, y, z, 1.0], dtype=np.float64)


def find_nearest_valid_depth_pixel(depth_img: np.ndarray, u: int, v: int, radius: int,
                                    max_depth_mm: int = DEPTH_MAX_VALID_MM):
    H, W = depth_img.shape[:2]
    if not (0 <= u < W and 0 <= v < H):
        return None
    best = None
    best_dist2 = None
    best_depth = None
    u0 = max(0, u - radius)
    u1 = min(W, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(H, v + radius + 1)
    for vv in range(v0, v1):
        for uu in range(u0, u1):
            d = int(depth_img[vv, uu])
            if d <= 0 or d >= max_depth_mm:
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


def get_place_anchor(
    depth_dir: Path,
    frame_index: int,
    ee_pose_arr: np.ndarray,
    T_cam_to_ee: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    u: int = CONTACT_U,
    v: int = CONTACT_V,
):
    """
    Extrae el place anchor 3D en frame base en el frame de release físico.
    Misma lógica que get_contact_anchor pero para place.
    """
    depth_file = depth_dir / f"{frame_index:06d}.png"
    if not depth_file.exists():
        raise FileNotFoundError(f"Depth file no encontrado: {depth_file}")

    depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise RuntimeError(f"No se pudo leer depth: {depth_file}")
    if depth_img.ndim != 2:
        raise ValueError(f"Depth debe ser monocanal. Shape: {depth_img.shape}")

    H, W = depth_img.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"Pixel ({u},{v}) fuera del shape depth {(H, W)}")

    depth_mm = int(depth_img[v, u])
    used_fallback = False

    if depth_mm <= 0 or depth_mm >= CONTACT_DEPTH_MAX_MM:
        nearest = find_nearest_valid_depth_pixel(
            depth_img, u, v, DEPTH_SEARCH_RADIUS, max_depth_mm=CONTACT_DEPTH_MAX_MM
        )
        if nearest is None:
            raise RuntimeError(
                f"Profundidad inválida en ({u},{v}) y no hay vecinos válidos "
                f"en radio {DEPTH_SEARCH_RADIUS}"
            )
        nb_u, nb_v, depth_mm = nearest
        used_fallback = True
    else:
        nb_u, nb_v = u, v

    depth_m = depth_mm / 1000.0

    if frame_index < 0 or frame_index >= len(ee_pose_arr):
        raise IndexError(
            f"frame_index {frame_index} fuera de rango para ee_pose len={len(ee_pose_arr)}"
        )

    # Si se usó fallback de depth, usar también el pixel vecino para XY
    # (consistente con la inferencia, que usa used_u/used_v para todo)
    p_cam = backproject(nb_u, nb_v, depth_m, fx, fy, cx, cy)

    T_ee_to_base = pose7_to_matrix(ee_pose_arr[frame_index])
    T_cam_to_base = T_ee_to_base @ T_cam_to_ee
    p_ee   = T_cam_to_ee  @ p_cam
    p_base = T_cam_to_base @ p_cam

    p_ee_xyz   = p_ee[:3].astype(np.float32)
    p_base_xyz = p_base[:3].astype(np.float32)

    debug_info = {
        "frame_index": int(frame_index),
        "depth_file": str(depth_file),
        "requested_pixel_uv": [int(u), int(v)],
        "used_pixel_uv": [int(nb_u), int(nb_v)],
        "used_fallback_depth_mm": bool(used_fallback),
        "depth_mm": int(depth_mm),
        "depth_m": float(depth_m),
        "point_cam_xyz": p_cam[:3].tolist(),
        "point_ee_xyz":  p_ee_xyz.tolist(),
        "point_base_xyz": p_base_xyz.tolist(),
    }

    return p_base_xyz, debug_info


# ── Detección del frame de place físico ─────────────────────────────────────

def detect_place_frame(g_trace: np.ndarray, cmd_trace: np.ndarray, **_) -> int:
    """
    Devuelve el frame en que se recibe el comando OPEN (primera transición
    close→open en cmd_trace). Este es el frame que el recorder captura como
    instante de place (_capture_place_pose), por lo que el anchor 3D se
    computa aquí — no en el instante de apertura física del gripper.
    Devuelve -1 si no se detecta ninguna transición.
    """
    cmd      = np.asarray(cmd_trace)
    open_val  = cmd.max()
    close_val = cmd.min()
    if open_val == close_val:
        return -1
    for i in range(1, len(cmd)):
        if cmd[i - 1] == close_val and cmd[i] == open_val:
            return i
    return -1


# ── Limpieza opcional de trayectorias ───────────────────────────────────────

def detect_motion_start(q: np.ndarray, threshold: float = 0.01, keep_before: int = 2) -> int:
    """
    Detecta el primer índice donde empieza movimiento articular real.

    dq[k] = ||q[k+1] - q[k]||_2
    Si dq[k] > threshold, consideramos que el movimiento empieza en k+1.
    Devuelve max(0, k+1-keep_before) para conservar un poco de contexto previo.
    """
    q = np.asarray(q, dtype=np.float64)
    if len(q) < 2:
        return 0

    dq = np.linalg.norm(np.diff(q, axis=0), axis=1)
    idxs = np.where(dq > threshold)[0]
    if len(idxs) == 0:
        return 0

    return max(0, int(idxs[0] + 1) - int(keep_before))


def build_clean_keep_indices(
    q: np.ndarray,
    g: np.ndarray,
    g_cmd: np.ndarray,
    start_idx: int,
    frame_index: int,
    q_eps: float = 1e-3,
    g_eps: float = 1e-3,
    release_keep_before: int = 3,
    release_keep_after: int = 8,
) -> np.ndarray:
    """
    Construye índices originales a conservar tras:
      - trim inicial
      - eliminación de frames consecutivos casi duplicados

    Siempre preserva:
      - start_idx
      - último frame
      - una ventana alrededor del frame de release
      - frames donde cambia el comando de gripper

    Nota: al eliminar frames estáticos, la acción absoluta pasa a ser q del
    siguiente frame conservado. Esto reduce targets q_next≈q_current que
    enseñan a la política a quedarse quieta.
    """
    n = len(q)
    if n == 0:
        return np.array([], dtype=np.int64)

    start_idx = int(np.clip(start_idx, 0, n - 1))
    frame_index = int(frame_index)

    mandatory = {start_idx, n - 1}

    # Preservar una ventana alrededor del release para no destruir la apertura.
    for k in range(frame_index - release_keep_before, frame_index + release_keep_after + 1):
        if start_idx <= k < n:
            mandatory.add(int(k))

    # Preservar cambios de comando de gripper.
    for i in range(start_idx + 1, n):
        if g_cmd[i] != g_cmd[i - 1]:
            mandatory.add(i - 1)
            mandatory.add(i)

    keep = [start_idx]
    for i in range(start_idx + 1, n):
        last = keep[-1]

        dq = float(np.max(np.abs(q[i] - q[last])))
        dg = float(abs(float(g[i]) - float(g[last])))
        cmd_changed = bool(g_cmd[i] != g_cmd[last])

        if i in mandatory or dq > q_eps or dg > g_eps or cmd_changed:
            keep.append(i)

    keep = sorted(set(keep).union(mandatory))
    keep = [i for i in keep if start_idx <= i < n]

    return np.asarray(keep, dtype=np.int64)


# ── Funciones base del conversor ────────────────────────────────────────────

def resize_and_encode(img_path: Path, size=(256, 256)) -> bytes:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {img_path}")
    if img.shape[:2] != (size[1], size[0]):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError(f"Error encoding: {img_path}")
    return encoded.tobytes()


def get_gripper_cmd_width(data):
    if "gripper_cmd" in data.files:
        cmd = data["gripper_cmd"].astype(np.float32)
    else:
        g = data["gripper"].astype(np.float32)
        cmd = (g > np.median(g)).astype(np.float32)
    return np.where(cmd > 0, OPEN_WIDTH, CLOSE_WIDTH).astype(np.float32)


def check_timestamps(data, ep_name):
    if "cam0_timestamps" not in data.files:
        print(f"[{ep_name}] ERROR: sin cam0_timestamps")
        return False
    ref = data["cam0_timestamps"].astype(np.float64)
    for key in SYNC_TIMESTAMP_KEYS:
        if key not in data.files:
            continue
        ts = data[key].astype(np.float64)
        n = min(len(ref), len(ts))
        diff = np.abs(ts[:n] - ref[:n]) * 1000.0
        print(
            f"[{ep_name}] {key:<18} "
            f"mean={diff.mean():6.1f}ms max={diff.max():6.1f}ms"
        )
        if diff.max() > 200.0:
            print(f"[{ep_name}] ADVERTENCIA: {key} tiene un desfase alto (>200ms)")
    return True


# ── Conversión principal ────────────────────────────────────────────────────

def convert_dataset(
    dataset_dir: str,
    out_path: str,
    hand_eye_yaml: str,
    img_size: int = 256,
    fx: float = D435I_FX,
    fy: float = D435I_FY,
    cx: float = D435I_CX,
    cy: float = D435I_CY,
    trim_start_motion: bool = False,
    motion_threshold: float = 0.01,
    keep_before_start: int = 2,
    dedup_static: bool = False,
    q_eps: float = 1e-3,
    g_eps: float = 1e-3,
    release_keep_before: int = 3,
    release_keep_after: int = 8,
):
    dataset_dir = Path(dataset_dir)
    episodes_dir = dataset_dir / "episodes"
    episode_dirs = sorted(episodes_dir.glob("episode_*"))

    T_cam_to_ee = load_T_cam_to_ee(Path(hand_eye_yaml))
    print(f"[INFO] T_cam_to_ee cargada desde {hand_eye_yaml}")
    print(f"[INFO] Intrinsics @ {img_size}px: fx={fx} fy={fy} cx={cx} cy={cy}")

    out_buffer    = []
    all_actions   = []
    all_contacts  = []

    skipped           = 0
    skipped_no_depth  = 0

    total_original_frames = 0
    total_kept_frames = 0
    total_trimmed_start_frames = 0
    total_dedup_removed_frames = 0

    size = (img_size, img_size)
    contact_debug = {}

    for ep in tqdm(episode_dirs, desc="Episodes"):
        traj_path     = ep / "traj.npz"
        cam0_dir      = ep / "cam0_256"
        cam1_dir      = ep / "cam1_256"
        cam0_depth_dir = ep / "cam0_depth"

        if not traj_path.exists():
            skipped += 1
            continue

        if not cam0_depth_dir.exists():
            print(f"[{ep.name}] SKIP — sin cam0_depth/")
            skipped += 1
            continue

        data = np.load(traj_path, allow_pickle=True)

        if "ee_pose" not in data.files:
            print(f"[{ep.name}] SKIP — sin ee_pose en traj.npz")
            skipped += 1
            continue

        if not check_timestamps(data, ep.name):
            skipped += 1
            continue

        if "gripper_cmd" not in data.files:
            print(f"[{ep.name}] SKIP — sin gripper_cmd en traj.npz")
            skipped += 1
            continue

        q     = data["q"].astype(np.float32)
        g     = data["gripper"].astype(np.float32)
        g_cmd = get_gripper_cmd_width(data)

        cam0_files = sorted(cam0_dir.glob("*.jpg"))
        cam1_files = sorted(cam1_dir.glob("*.jpg"))

        t = min(len(q), len(g), len(cam0_files), len(cam1_files), len(data["ee_pose"]))

        q     = q[:t]
        g     = g[:t]
        g_cmd = g_cmd[:t]
        cam0_files = cam0_files[:t]
        cam1_files = cam1_files[:t]

        try:
            ee_pose_clean = sanitize_ee_pose(data["ee_pose"][:t], ep.name)
        except RuntimeError as e:
            print(f"[{ep.name}] SKIP — {e}")
            skipped += 1
            continue

        total_original_frames += int(t)

        if t < 2:
            print(f"[{ep.name}] SKIP — episodio demasiado corto (t={t})")
            skipped += 1
            continue

        # Detectar frame de release físico (gripper abre con el objeto)
        frame_index = detect_place_frame(g, data["gripper_cmd"][:t])
        if frame_index < 0:
            print(f"[{ep.name}] SKIP — no se detecta release físico")
            skipped += 1
            continue

        try:
            place_anchor, debug_info = get_place_anchor(
                depth_dir=cam0_depth_dir,
                frame_index=frame_index,
                ee_pose_arr=ee_pose_clean,
                T_cam_to_ee=T_cam_to_ee,
                fx=fx, fy=fy, cx=cx, cy=cy,
                u=CONTACT_U, v=CONTACT_V,
            )
        except Exception as e:
            print(f"[{ep.name}] SKIP — error place anchor: {e}")
            skipped_no_depth += 1
            skipped += 1
            continue

        debug_info["episode_name"] = ep.name
        contact_debug[ep.name] = debug_info

        p_cam_xyz = debug_info["point_cam_xyz"]
        p_ee_xyz  = debug_info["point_ee_xyz"]
        print(
            f"[{ep.name}] place_anchor "
            f"(cam) = {np.round(p_cam_xyz, 4)} m | "
            f"(EE)  = {np.round(p_ee_xyz,  4)} m | "
            f"(base)= {np.round(place_anchor, 4).tolist()} m "
            f"(frame {frame_index}, depth={debug_info['depth_mm']}mm)"
        )

        ee_pose_arr = ee_pose_clean

        # NO-SPLIT + limpieza opcional:
        # Cada episodio sigue siendo UNA trayectoria, pero podemos:
        #   1) recortar la pausa inicial antes de que el brazo empiece a moverse
        #   2) eliminar frames consecutivos casi duplicados que generan action_q≈state_q
        start_idx = 0
        if trim_start_motion:
            start_idx = detect_motion_start(
                q,
                threshold=motion_threshold,
                keep_before=keep_before_start,
            )

        if frame_index < start_idx:
            print(
                f"[{ep.name}] SKIP — frame release ({frame_index}) queda antes "
                f"del start_idx ({start_idx})"
            )
            skipped += 1
            continue

        if dedup_static:
            keep_idx = build_clean_keep_indices(
                q=q,
                g=g,
                g_cmd=g_cmd,
                start_idx=start_idx,
                frame_index=frame_index,
                q_eps=q_eps,
                g_eps=g_eps,
                release_keep_before=release_keep_before,
                release_keep_after=release_keep_after,
            )
        else:
            keep_idx = np.arange(start_idx, t, dtype=np.int64)

        if len(keep_idx) < 2:
            print(f"[{ep.name}] SKIP — trayectoria demasiado corta tras limpieza")
            skipped += 1
            continue

        trimmed = int(start_idx)
        removed = int(t - len(keep_idx) - trimmed)
        total_trimmed_start_frames += trimmed
        total_dedup_removed_frames += max(0, removed)
        total_kept_frames += int(len(keep_idx))

        if trim_start_motion or dedup_static:
            print(
                f"[{ep.name}] clean: original={t} start_idx={start_idx} "
                f"kept={len(keep_idx)} trimmed_start={trimmed} "
                f"dedup_removed={max(0, removed)} release_frame={frame_index}"
            )

        segments = [{
            "q": q[keep_idx],
            "g": g[keep_idx],
            "g_cmd": g_cmd[keep_idx],
            "cam0_files": [cam0_files[int(i)] for i in keep_idx],
            "cam1_files": [cam1_files[int(i)] for i in keep_idx],
            "ee_pose": ee_pose_arr[keep_idx],
            "orig_idx": keep_idx,
        }]

        # PLACE always-world-fixed: el target en base frame se computa
        # en get_place_anchor() y se re-expresa en EE actual en cada frame.
        # Nota: place_anchor ya está en float32 (p_base_xyz de get_place_anchor).
        p_base_h_anchor = np.append(place_anchor.astype(np.float64), 1.0)  # (4,)

        ep_had_valid_segment = False

        for seg in segments:
            q_s       = seg["q"]
            g_s       = seg["g"]
            g_cmd_s   = seg["g_cmd"]
            cam0_s    = seg["cam0_files"]
            cam1_s    = seg["cam1_files"]
            ee_pose_s = seg["ee_pose"]

            T_seg = len(q_s) - 1

            state   = np.concatenate([q_s, g_s[:, None]], axis=1)
            actions = np.concatenate([q_s[1:], g_cmd_s[1:, None]], axis=1)

            orig_idx_s = seg["orig_idx"]
            traj_out = []
            for i in range(T_seg):
                # PLACE always-world-fixed:
                # El target de place está fijo en base/world durante TODO el episodio.
                # En cada frame lo reexpresamos en el EE actual, sin freeze post-open.
                # Semántica limpia: cuando el vector → 0, el EE está sobre el target.
                T_ee_i_to_base = pose7_to_matrix(ee_pose_s[i])
                T_base_to_ee_i = np.linalg.inv(T_ee_i_to_base)
                p_ee_i = (T_base_to_ee_i @ p_base_h_anchor)[:3].astype(np.float32)

                obs = {
                    "state": state[i],
                    "enc_cam_0": resize_and_encode(cam0_s[i], size),
                    "enc_cam_1": resize_and_encode(cam1_s[i], size),
                    "contact_anchor": p_ee_i,
                }
                traj_out.append((obs, actions[i], 0.0))

            if len(traj_out) == 0:
                continue

            out_buffer.append(traj_out)
            all_actions.append(actions)
            all_contacts.extend([obs["contact_anchor"] for obs, _, _ in traj_out])
            ep_had_valid_segment = True

        if not ep_had_valid_segment:
            skipped += 1
            continue

    if len(out_buffer) == 0:
        raise RuntimeError(
            "No hay episodios válidos. "
            "Todos fueron filtrados por timestamps, datos faltantes o depth inválido."
        )

    actions_all  = np.concatenate(all_actions, axis=0)
    contacts_all = np.stack(all_contacts, axis=0)

    # Normalización acciones [-1, 1]
    ac_min = np.percentile(actions_all, 1, axis=0)
    ac_max = np.percentile(actions_all, 99, axis=0)
    loc   = (ac_min + ac_max) / 2.0
    scale = (ac_max - ac_min).clip(min=1e-6) / 2.0

    # Normalización place anchor
    c_min = np.percentile(contacts_all, 1, axis=0)
    c_max = np.percentile(contacts_all, 99, axis=0)
    contact_loc   = ((c_min + c_max) / 2.0).astype(np.float32)
    contact_scale = ((c_max - c_min) / 2.0).clip(min=1e-6).astype(np.float32)

    for traj in out_buffer:
        for i, (obs, a, r) in enumerate(traj):
            a = np.clip((a - loc) / scale, -1, 1).astype(np.float32)
            ca_norm = np.clip(
                (obs["contact_anchor"] - contact_loc) / contact_scale, -1, 1
            ).astype(np.float32)
            obs["contact_anchor"] = ca_norm
            traj[i] = (obs, a, r)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "buf.pkl", "wb") as f:
        pkl.dump(out_buffer, f)

    with open(out_path / "ac_norm.json", "w") as f:
        json.dump({"loc": loc.tolist(), "scale": scale.tolist()}, f, indent=2)

    with open(out_path / "contact_norm.json", "w") as f:
        json.dump({
            "loc":   contact_loc.tolist(),
            "scale": contact_scale.tolist(),
        }, f, indent=2)

    with open(out_path / "contact_debug.json", "w") as f:
        json.dump(contact_debug, f, indent=2)

    print("\nDONE")
    print("episodes:", len(out_buffer))
    print("frames:", len(actions_all))
    print("skipped:", skipped)
    print("  of which no valid depth/anchor:", skipped_no_depth)
    print("mode: no_split")
    print("trim_start_motion:", trim_start_motion)
    print("dedup_static:", dedup_static)
    print("original frames:", total_original_frames)
    print("kept frames after cleaning:", total_kept_frames if (trim_start_motion or dedup_static) else len(actions_all) + len(out_buffer))
    print("trimmed start frames:", total_trimmed_start_frames)
    print("dedup removed frames:", total_dedup_removed_frames)
    print("place_anchor_loc:  ", np.round(contact_loc, 4).tolist())
    print("place_anchor_scale:", np.round(contact_scale, 4).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",   required=True)
    parser.add_argument("--out_path",      required=True)
    parser.add_argument("--hand_eye_yaml", default="my_data/hand_eye_result_change.yaml")
    parser.add_argument("--img_size",      type=int,   default=256)
    parser.add_argument("--fx",            type=float, default=D435I_FX)
    parser.add_argument("--fy",            type=float, default=D435I_FY)
    parser.add_argument("--cx",            type=float, default=D435I_CX)
    parser.add_argument("--cy",            type=float, default=D435I_CY)

    # Ablation / cleaning options
    parser.add_argument(
        "--trim_start_motion",
        action="store_true",
        help="Recorta la pausa inicial hasta el primer movimiento articular real, conservando --keep_before_start frames.",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=0.01,
        help="Umbral L2 de ||q[i+1]-q[i]|| para detectar inicio de movimiento.",
    )
    parser.add_argument(
        "--keep_before_start",
        type=int,
        default=2,
        help="Frames a conservar antes del primer movimiento detectado.",
    )
    parser.add_argument(
        "--dedup_static",
        action="store_true",
        help="Elimina frames consecutivos casi duplicados para reducir acciones q_next≈q_current.",
    )
    parser.add_argument(
        "--q_eps",
        type=float,
        default=1e-3,
        help="Umbral max|Δq| para considerar que dos frames consecutivos son distintos.",
    )
    parser.add_argument(
        "--g_eps",
        type=float,
        default=1e-3,
        help="Umbral |Δgripper| para conservar un frame aunque q no cambie.",
    )
    parser.add_argument(
        "--release_keep_before",
        type=int,
        default=3,
        help="Frames originales a preservar antes del frame de release cuando se usa --dedup_static.",
    )
    parser.add_argument(
        "--release_keep_after",
        type=int,
        default=8,
        help="Frames originales a preservar después del frame de release cuando se usa --dedup_static.",
    )

    args = parser.parse_args()

    convert_dataset(
        dataset_dir=args.dataset_dir,
        out_path=args.out_path,
        hand_eye_yaml=args.hand_eye_yaml,
        img_size=args.img_size,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        trim_start_motion=args.trim_start_motion,
        motion_threshold=args.motion_threshold,
        keep_before_start=args.keep_before_start,
        dedup_static=args.dedup_static,
        q_eps=args.q_eps,
        g_eps=args.g_eps,
        release_keep_before=args.release_keep_before,
        release_keep_after=args.release_keep_after,
    )


if __name__ == "__main__":
    main()
