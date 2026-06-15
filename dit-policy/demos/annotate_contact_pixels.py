#!/usr/bin/env python3
"""
annotate_contact_pixels.py

Script para seleccionar manualmente el pixel de contacto en el frame de grasp/cierre
de cada episodio grabado con recorder_node_two_cam.py.

Entrada esperada:
    ~/dit_demos/pick_demo/
      episodes/
        episode_0000/
          cam0_256/
          cam1_256/
          traj.npz
          grasp_poses.json      # opcional

Salida por episodio:
    episode_0000/contact_pixel.json

Formato guardado:
    {
      "camera": "cam0",
      "frame_index": 42,
      "u": 160,
      "v": 110,
      "image_file": "cam0_256/000042.jpg",
      "selection_source": "manual",
      "frame_mode": "closed_after_cmd",
      "gripper_cmd": -1,
      "gripper_measured": 0.0048
    }

Uso recomendado:
    python annotate_contact_pixels.py \
        --dataset_dir ~/dit_demos/pick_demo \
        --camera cam0 \
        --default-u 160 \
        --default-v 110 \
        --frame-mode closed_after_cmd

Controles GUI:
    click izquierdo  -> seleccionar pixel
    Enter / s / c    -> guardar y pasar al siguiente episodio
    a / izquierda    -> frame anterior
    d / derecha      -> frame siguiente
    r                -> reset al pixel por defecto
    n                -> saltar episodio
    q / Esc          -> salir
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


OPEN_CMD = 1
CLOSE_CMD = -1


def as_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def get_camera_dir(ep: Path, camera: str) -> Path:
    if camera == "cam0":
        return ep / "cam0_256"
    if camera == "cam1":
        return ep / "cam1_256"
    raise ValueError(f"camera no soportada: {camera}")


def first_close_command_frame(data) -> Optional[int]:
    """
    Primer índice donde gripper_cmd pasa a cerrado.
    """
    if "gripper_cmd" not in data.files:
        return None

    cmd = data["gripper_cmd"].astype(np.int32)
    close_idx = np.where(cmd < 0)[0]
    if len(close_idx) == 0:
        return None

    return int(close_idx[0])


def closed_after_cmd_frame(
    data,
    delay_frames: int = 3,
    closed_threshold: Optional[float] = None,
) -> Optional[int]:
    """
    Intenta elegir un frame donde el gripper ya está físicamente cerrado,
    no solo comandado a cerrar.

    Estrategia:
      1. Encuentra primer gripper_cmd < 0.
      2. Si existe 'gripper', busca después de ese índice el primer frame
         donde gripper <= closed_threshold.
      3. Si no hay threshold, usa percentil bajo del gripper medido.
      4. Si no puede, usa close_cmd_idx + delay_frames.

    Nota:
      En tu recorder 'gripper' es sum(msg.position), así que suele ser mayor
      cuando está abierto y menor cuando está cerrado.
    """
    idx_cmd = first_close_command_frame(data)
    if idx_cmd is None:
        return None

    n = len(data["gripper_cmd"])

    if "gripper" not in data.files:
        return int(min(idx_cmd + delay_frames, n - 1))

    g = data["gripper"].astype(np.float32)
    n = min(n, len(g))

    if closed_threshold is None:
        # Umbral robusto: cerca de los valores cerrados.
        # Si hay mucha variación, el percentil 20 suele estar en zona cerrada.
        closed_threshold = float(np.percentile(g[:n], 20))

    search_start = min(idx_cmd + 1, n - 1)
    candidates = np.where(g[search_start:n] <= closed_threshold)[0]

    if len(candidates) > 0:
        return int(search_start + candidates[0])

    return int(min(idx_cmd + delay_frames, n - 1))


def grasp_pose_frame(ep: Path) -> Optional[int]:
    """
    Lee grasp_poses.json si existe. Ojo: en tu recorder se guarda cuando se
    detecta el comando de cierre, no necesariamente cuando ya terminó de cerrar.
    """
    p = ep / "grasp_poses.json"
    data = load_json(p)
    if not data:
        return None

    if isinstance(data, list) and len(data) > 0:
        if "frame_index" in data[0]:
            return int(data[0]["frame_index"])

    return None


def select_initial_frame(
    ep: Path,
    traj,
    frame_mode: str,
    delay_frames: int,
    closed_threshold: Optional[float],
) -> int:
    n = len(traj["q"]) if "q" in traj.files else len(traj["gripper_cmd"])

    if frame_mode == "closed_after_cmd":
        idx = closed_after_cmd_frame(
            traj,
            delay_frames=delay_frames,
            closed_threshold=closed_threshold,
        )
    elif frame_mode == "command_close":
        idx = first_close_command_frame(traj)
    elif frame_mode == "grasp_poses":
        idx = grasp_pose_frame(ep)
    elif frame_mode == "manual_start":
        idx = 0
    else:
        raise ValueError(f"frame_mode no soportado: {frame_mode}")

    if idx is None:
        idx = 0

    return clamp_int(idx, 0, n - 1)


class ContactAnnotator:
    def __init__(
        self,
        ep: Path,
        camera: str,
        frame_index: int,
        default_u: int,
        default_v: int,
        frame_mode: str,
        closed_threshold: Optional[float],
        overwrite: bool,
    ):
        self.ep = ep
        self.camera = camera
        self.camera_dir = get_camera_dir(ep, camera)
        self.image_files = sorted(self.camera_dir.glob("*.jpg"))
        if not self.image_files:
            raise FileNotFoundError(f"No hay imágenes en {self.camera_dir}")

        self.depth_dir = ep / "cam0_depth"
        self.depth_files = sorted(self.depth_dir.glob("*.png")) if self.depth_dir.exists() else []

        self.traj = np.load(ep / "traj.npz", allow_pickle=True)
        self.frame_index = clamp_int(frame_index, 0, len(self.image_files) - 1)
        self.default_u = int(default_u)
        self.default_v = int(default_v)
        self.u = int(default_u)
        self.v = int(default_v)
        self.frame_mode = frame_mode
        self.closed_threshold = closed_threshold
        self.overwrite = overwrite
        self.window = f"{ep.name} - {camera}"

        existing = ep / "contact_pixel.json"
        if existing.exists() and not overwrite:
            old = load_json(existing)
            if old is not None:
                self.u = int(old.get("u", self.u))
                self.v = int(old.get("v", self.v))
                self.frame_index = int(old.get("frame_index", self.frame_index))

    def current_image_path(self) -> Path:
        return self.image_files[self.frame_index]

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.u = int(x)
            self.v = int(y)

    def _get_depth_image(self) -> Optional[np.ndarray]:
        if not self.depth_files:
            return None
        fi = min(self.frame_index, len(self.depth_files) - 1)
        d = cv2.imread(str(self.depth_files[fi]), cv2.IMREAD_ANYDEPTH)
        return d

    def draw(self):
        img = cv2.imread(str(self.current_image_path()), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer {self.current_image_path()}")

        h, w = img.shape[:2]
        u = clamp_int(self.u, 0, w - 1)
        v = clamp_int(self.v, 0, h - 1)

        vis = img.copy()

        # Depth overlay: red tint where depth == 0 (invalid)
        depth_img = self._get_depth_image()
        depth_at_cursor = None
        if depth_img is not None:
            dh, dw = depth_img.shape[:2]
            if dh == h and dw == w:
                zero_mask = (depth_img == 0)
                overlay = vis.copy()
                overlay[zero_mask] = (0, 0, 200)
                vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
                du = clamp_int(u, 0, dw - 1)
                dv = clamp_int(v, 0, dh - 1)
                depth_at_cursor = int(depth_img[dv, du])

        # Crosshair — yellow if valid depth, red if invalid
        cursor_color = (0, 255, 0) if (depth_at_cursor is not None and depth_at_cursor > 0) else (0, 0, 255)
        cv2.drawMarker(vis, (u, v), cursor_color, markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.circle(vis, (u, v), 5, cursor_color, 2)

        # Texto
        gcmd = self.get_gripper_cmd_for_frame()
        gmeas = self.get_gripper_measured_for_frame()
        depth_str = f"{depth_at_cursor} mm" if depth_at_cursor is not None else "N/A"
        depth_valid = depth_at_cursor is not None and depth_at_cursor > 0

        lines = [
            f"{self.ep.name} | frame {self.frame_index}/{len(self.image_files)-1}",
            f"pixel: u={u}, v={v} | depth={depth_str} ({'OK' if depth_valid else 'INVALID - click elsewhere!'})",
            f"gripper_cmd={gcmd} | gripper_measured={gmeas}",
            "click=select | Enter/s/c=save | a/d=frame | r=reset | n=skip | q=quit",
            "[RED TINT = zero depth — pick a pixel WITHOUT red overlay]",
        ]

        y0 = 24
        for i, line in enumerate(lines):
            color = (0, 200, 255) if i == 4 else (255, 255, 255)
            cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
            y0 += 20

        return vis

    def get_gripper_cmd_for_frame(self):
        if "gripper_cmd" not in self.traj.files:
            return None
        arr = self.traj["gripper_cmd"]
        if self.frame_index >= len(arr):
            return None
        return int(arr[self.frame_index])

    def get_gripper_measured_for_frame(self):
        if "gripper" not in self.traj.files:
            return None
        arr = self.traj["gripper"]
        if self.frame_index >= len(arr):
            return None
        return float(arr[self.frame_index])

    def save(self):
        img_rel = self.current_image_path().relative_to(self.ep)

        out = {
            "camera": self.camera,
            "frame_index": int(self.frame_index),
            "u": int(self.u),
            "v": int(self.v),
            "image_file": str(img_rel),
            "selection_source": "manual",
            "frame_mode": self.frame_mode,
            "default_u": int(self.default_u),
            "default_v": int(self.default_v),
            "closed_threshold": self.closed_threshold,
            "gripper_cmd": self.get_gripper_cmd_for_frame(),
            "gripper_measured": self.get_gripper_measured_for_frame(),
        }

        out_path = self.ep / "contact_pixel.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"[OK] guardado: {out_path}")

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse_cb)

        while True:
            vis = self.draw()
            cv2.imshow(self.window, vis)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 10, ord("s"), ord("c")):  # Enter / save / confirm
                self.save()
                cv2.destroyWindow(self.window)
                return "saved"

            if key in (ord("n"),):
                print(f"[SKIP] {self.ep.name}")
                cv2.destroyWindow(self.window)
                return "skip"

            if key in (ord("q"), 27):  # q or Esc
                cv2.destroyWindow(self.window)
                return "quit"

            if key in (ord("a"), 81, 2):  # left
                self.frame_index = clamp_int(self.frame_index - 1, 0, len(self.image_files) - 1)

            if key in (ord("d"), 83, 3):  # right
                self.frame_index = clamp_int(self.frame_index + 1, 0, len(self.image_files) - 1)

            if key == ord("r"):
                self.u = self.default_u
                self.v = self.default_v


def auto_save_default(
    ep: Path,
    camera: str,
    frame_index: int,
    default_u: int,
    default_v: int,
    frame_mode: str,
    closed_threshold: Optional[float],
    overwrite: bool,
):
    out_path = ep / "contact_pixel.json"
    if out_path.exists() and not overwrite:
        print(f"[KEEP] {out_path} ya existe")
        return

    cam_dir = get_camera_dir(ep, camera)
    image_file = cam_dir / f"{frame_index:06d}.jpg"
    if not image_file.exists():
        files = sorted(cam_dir.glob("*.jpg"))
        if not files:
            print(f"[SKIP] {ep.name}: no hay imágenes")
            return
        frame_index = min(frame_index, len(files) - 1)
        image_file = files[frame_index]

    traj = np.load(ep / "traj.npz", allow_pickle=True)

    gcmd = int(traj["gripper_cmd"][frame_index]) if "gripper_cmd" in traj.files and frame_index < len(traj["gripper_cmd"]) else None
    gmeas = float(traj["gripper"][frame_index]) if "gripper" in traj.files and frame_index < len(traj["gripper"]) else None

    out = {
        "camera": camera,
        "frame_index": int(frame_index),
        "u": int(default_u),
        "v": int(default_v),
        "image_file": str(image_file.relative_to(ep)),
        "selection_source": "auto_default",
        "frame_mode": frame_mode,
        "default_u": int(default_u),
        "default_v": int(default_v),
        "closed_threshold": closed_threshold,
        "gripper_cmd": gcmd,
        "gripper_measured": gmeas,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] auto guardado: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Ej: ~/dit_demos/pick_demo")
    parser.add_argument("--camera", choices=["cam0", "cam1"], default="cam0")
    parser.add_argument("--default-u", type=int, default=160)
    parser.add_argument("--default-v", type=int, default=110)
    parser.add_argument(
        "--frame-mode",
        choices=["closed_after_cmd", "command_close", "grasp_poses", "manual_start"],
        default="closed_after_cmd",
        help="Cómo elegir el frame inicial mostrado.",
    )
    parser.add_argument(
        "--delay-frames",
        type=int,
        default=3,
        help="Fallback: frames después del comando de cierre si no se puede detectar cierre físico.",
    )
    parser.add_argument(
        "--closed-threshold",
        type=float,
        default=None,
        help="Umbral de gripper medido para considerar cerrado. Si no se da, usa percentil 20 por episodio.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe contact_pixel.json existente.")
    parser.add_argument("--auto", action="store_true", help="No abre GUI; guarda el pixel default en el frame elegido.")
    parser.add_argument("--fix_invalid_depth", action="store_true",
                        help="Solo procesa episodios donde el depth en el pixel anotado (o el default) es inválido (0). "
                             "Implica --overwrite para esos episodios.")

    args = parser.parse_args()

    dataset_dir = as_path(args.dataset_dir)
    episodes_dir = dataset_dir / "episodes"
    episode_dirs = sorted([p for p in episodes_dir.glob("episode_*") if p.is_dir()])

    if not episode_dirs:
        raise FileNotFoundError(f"No encontré episodios en {episodes_dir}")

    print(f"Dataset: {dataset_dir}")
    print(f"Episodios: {len(episode_dirs)}")
    print(f"Pixel inicial: ({args.default_u}, {args.default_v})")
    print(f"Frame mode: {args.frame_mode}")

    for ep in episode_dirs:
        out_path = ep / "contact_pixel.json"

        if args.fix_invalid_depth:
            # Only process episodes where depth at the annotated pixel is invalid.
            # Use the same frame the converter will use (contact frame), not frame 0.
            traj_path_pre = ep / "traj.npz"
            if out_path.exists():
                with open(out_path) as _f:
                    _pj = json.load(_f)
                pix_u = int(_pj.get("u", args.default_u))
                pix_v = int(_pj.get("v", args.default_v))
                # frame_index in JSON = what converter will use
                if "frame_index" in _pj and _pj["frame_index"] is not None:
                    fi = int(_pj["frame_index"])
                elif traj_path_pre.exists():
                    _t = np.load(traj_path_pre, allow_pickle=True)
                    fi = select_initial_frame(ep, _t, args.frame_mode, args.delay_frames, args.closed_threshold)
                else:
                    fi = 0
            else:
                pix_u, pix_v = args.default_u, args.default_v
                # No JSON → use same contact-frame detection as converter
                if traj_path_pre.exists():
                    _t = np.load(traj_path_pre, allow_pickle=True)
                    fi = select_initial_frame(ep, _t, args.frame_mode, args.delay_frames, args.closed_threshold)
                else:
                    fi = 0
            depth_dir = ep / "cam0_depth"
            depth_files = sorted(depth_dir.glob("*.png")) if depth_dir.exists() else []
            if depth_files:
                fi = min(fi, len(depth_files) - 1)
                depth_img = cv2.imread(str(depth_files[fi]), cv2.IMREAD_ANYDEPTH)
                if depth_img is not None and depth_img[pix_v, pix_u] > 0:
                    print(f"[KEEP] {ep.name}: depth válido ({depth_img[pix_v, pix_u]} mm) en ({pix_u},{pix_v}) frame={fi}")
                    continue
            print(f"[FIX]  {ep.name}: depth inválido en ({pix_u},{pix_v}) frame={fi} — abriendo GUI")
        elif out_path.exists() and not args.overwrite:
            print(f"[KEEP] {ep.name}: ya existe contact_pixel.json")
            continue

        traj_path = ep / "traj.npz"
        if not traj_path.exists():
            print(f"[SKIP] {ep.name}: no existe traj.npz")
            continue

        cam_dir = get_camera_dir(ep, args.camera)
        if not cam_dir.exists():
            print(f"[SKIP] {ep.name}: no existe {cam_dir}")
            continue

        traj = np.load(traj_path, allow_pickle=True)
        frame_index = select_initial_frame(
            ep=ep,
            traj=traj,
            frame_mode=args.frame_mode,
            delay_frames=args.delay_frames,
            closed_threshold=args.closed_threshold,
        )

        if args.auto:
            auto_save_default(
                ep=ep,
                camera=args.camera,
                frame_index=frame_index,
                default_u=args.default_u,
                default_v=args.default_v,
                frame_mode=args.frame_mode,
                closed_threshold=args.closed_threshold,
                overwrite=args.overwrite,
            )
            continue

        annotator = ContactAnnotator(
            ep=ep,
            camera=args.camera,
            frame_index=frame_index,
            default_u=args.default_u,
            default_v=args.default_v,
            frame_mode=args.frame_mode,
            closed_threshold=args.closed_threshold,
            overwrite=args.overwrite,
        )
        result = annotator.run()
        if result == "quit":
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
