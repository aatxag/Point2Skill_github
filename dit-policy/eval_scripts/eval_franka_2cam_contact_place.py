#!/usr/bin/env python3
"""
FR3 eval script — contact/place-conditioned diffusion policy, two cameras.

Workflow per rollout:
  1. User clicks the contact pixel on the live 256×256 wrist image.
  2. Depth at that pixel is read and back-projected to the robot base frame
     using the D435i intrinsics (scaled to 256×256) and hand-eye calibration —
     identical to the training converter.
  3. During the rollout each step:
       pre-grasp  → anchor expressed in current EE frame (vector shrinks toward 0)
       post-grasp → anchor frozen to its value at the moment gripper confirms closed

Usage:
  python eval_franka_2cam_contact.py path/to/checkpoint.ckpt \\
      --hand_eye my_data/hand_eye_result.yaml
"""

import argparse
import json
import os
import pickle
import time
from collections import deque
from pathlib import Path

import cv2
import hydra
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from eval_franka_env_2cam_contact import make_fr3_env_2cam_contact

torch.set_float32_matmul_precision("high")

# ── FR3 joint limits (8-dim: 7 joints + gripper) ─────────────────────────────
FR3_JOINT_LIMITS_LOW = np.array(
    [-2.3093, -1.5133, -2.4937, -3.0500, -2.4800, 0.8521, -2.6895, 0.0],
    dtype=np.float32,
)
FR3_JOINT_LIMITS_HIGH = np.array(
    [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895, 0.08],
    dtype=np.float32,
)

# ── Depth constants (identical to training converter) ─────────────────────────
DEPTH_MAX_VALID_MM = 5000
DEPTH_SEARCH_RADIUS = 5

# ── D435i intrinsics scaled to 256×256 (identical to training) ───────────────
# Source: ros2 topic echo /camera/camera_wrist/color/camera_info @ 1280×720
#   sx = 256/1280 = 0.200,  sy = 256/720 ≈ 0.3556
D435I_FX = 181.685
D435I_FY = 322.924
D435I_CX = 129.035
D435I_CY = 131.825


# ── Geometry helpers ──────────────────────────────────────────────────────────

def load_T_cam_to_ee(yaml_path: str) -> np.ndarray:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    T = np.array(data["T_cam_to_ee_4x4"], dtype=np.float64)
    assert T.shape == (4, 4), f"Expected (4,4), got {T.shape}"
    return T


def find_valid_depth(depth_img: np.ndarray, u: int, v: int, radius: int):
    """Nearest pixel with 0 < depth < DEPTH_MAX_VALID_MM within radius."""
    H, W = depth_img.shape[:2]
    best, best_d2 = None, float("inf")
    for vv in range(max(0, v - radius), min(H, v + radius + 1)):
        for uu in range(max(0, u - radius), min(W, u + radius + 1)):
            d = int(depth_img[vv, uu])
            if d <= 0 or d >= DEPTH_MAX_VALID_MM:
                continue
            d2 = (uu - u) ** 2 + (vv - v) ** 2
            if d2 < best_d2:
                best, best_d2 = (uu, vv, d), d2
    return best


def backproject(u: int, v: int, depth_m: float) -> np.ndarray:
    """Pixel + depth → homogeneous point in camera frame (4,)."""
    return np.array([
        (u - D435I_CX) * depth_m / D435I_FX,
        (v - D435I_CY) * depth_m / D435I_FY,
        depth_m,
        1.0,
    ], dtype=np.float64)


# ── Click UI ─────────────────────────────────────────────────────────────────

# Colores tema blanco
_W_BG      = "#ffffff"   # fondo principal
_W_PANEL   = "#f4f6f9"   # panel de info
_W_ACCENT  = "#1565C0"   # azul brand (logo)
_W_ACCENT2 = "#1e88e5"   # azul claro para crosshair
_W_GREEN   = "#2e7d32"   # depth válido
_W_RED     = "#c62828"   # marker / depth inválido
_W_AMBER   = "#e65100"   # fallback
_W_DIM     = "#90a4ae"   # labels secundarios
_W_TEXT    = "#1a237e"   # texto principal
_W_BORDER  = "#cfd8dc"   # bordes

_LOGO_PATH = Path(__file__).parent.parent / "my_data" / "POINT2SKILL_LOGO.png"


def _load_logo():
    """Carga el logo Point2Skill como RGB."""
    if not _LOGO_PATH.exists():
        return None
    img = cv2.imread(str(_LOGO_PATH), cv2.IMREAD_COLOR)
    return img[:, :, ::-1] if img is not None else None   # BGR → RGB


def pick_contact_pixel(rgb_256: np.ndarray, depth_256: np.ndarray):
    """
    UI blanca con logo Point2Skill para seleccionar el pixel de contacto.
    Panel izquierdo: imagen + crosshair en vivo.
    Panel derecho: cursor, depth y punto seleccionado.
    Se cierra 1 s después del click.
    Devuelve (u, v) en coordenadas 256×256, o None si se cierra sin click.
    """
    rgb_display = rgb_256[:, :, ::-1].copy()   # BGR → RGB
    valid_mask  = (depth_256 > 0) & (depth_256 < DEPTH_MAX_VALID_MM)
    pct         = 100.0 * valid_mask.mean()
    clicked     = []
    logo_rgb    = _load_logo()

    fig = plt.figure(figsize=(11, 7.4), facecolor=_W_BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    HEADER_TOP    = 0.985
    HEADER_BOTTOM = 0.80
    DIVIDER_Y     = HEADER_BOTTOM - 0.006

    # ── Logo (fondo blanco, se ve perfecto) ──────────────────────────────
    if logo_rgb is not None:
        ax_logo = fig.add_axes([0.012, HEADER_BOTTOM + 0.010,
                                0.40,  HEADER_TOP - HEADER_BOTTOM - 0.014])
        ax_logo.set_facecolor(_W_BG)
        ax_logo.imshow(logo_rgb, aspect="auto",
                       extent=[0, logo_rgb.shape[1], logo_rgb.shape[0], 0],
                       interpolation="lanczos")
        ax_logo.set_xlim(0, logo_rgb.shape[1])
        ax_logo.set_ylim(logo_rgb.shape[0], 0)
        ax_logo.axis("off")
    else:
        fig.text(0.025, (HEADER_TOP + HEADER_BOTTOM) / 2,
                 "Point2Skill", ha="left", va="center",
                 fontsize=22, fontweight="bold", color=_W_ACCENT)

    # ── Depth coverage (derecha del header) ──────────────────────────────
    pct_col = _W_GREEN if pct > 50 else _W_AMBER
    fig.text(0.985, (HEADER_TOP + HEADER_BOTTOM) / 2 + 0.015,
             f"{pct:.0f}%", ha="right", va="center",
             fontsize=18, fontweight="bold", color=pct_col,
             fontfamily="monospace")
    fig.text(0.985, (HEADER_TOP + HEADER_BOTTOM) / 2 - 0.032,
             "depth coverage", ha="right", va="center",
             fontsize=7.5, color=_W_DIM, fontfamily="monospace")

    # ── Línea divisora azul ───────────────────────────────────────────────
    for lw, alpha in [(4.0, 0.12), (1.2, 1.0)]:
        fig.add_artist(mpatches.FancyArrowPatch(
            (0.0, DIVIDER_Y), (1.0, DIVIDER_Y),
            arrowstyle="-", color=_W_ACCENT, linewidth=lw,
            alpha=alpha, transform=fig.transFigure, clip_on=False,
        ))

    # ── Panel izquierdo: imagen ───────────────────────────────────────────
    IMG_BOTTOM = 0.045
    IMG_HEIGHT = HEADER_BOTTOM - IMG_BOTTOM - 0.025
    ax = fig.add_axes([0.015, IMG_BOTTOM, 0.595, IMG_HEIGHT])
    ax.set_facecolor(_W_BG)
    ax.imshow(rgb_display, extent=[0, 255, 255, 0], aspect="auto",
              interpolation="lanczos")
    ax.set_xlim(0, 255); ax.set_ylim(255, 0)
    for sp in ax.spines.values():
        sp.set_edgecolor(_W_BORDER); sp.set_linewidth(1.2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(2, 253, "256×256", fontsize=6, color=_W_DIM,
            fontfamily="monospace", va="bottom")

    # Crosshair en vivo
    hline = ax.axhline(-999, color=_W_ACCENT2, linewidth=0.8,
                        alpha=0.7, linestyle="--")
    vline = ax.axvline(-999, color=_W_ACCENT2, linewidth=0.8,
                        alpha=0.7, linestyle="--")

    # Marcador del click: halo + anillo + cruz
    ring_outer = ax.plot([], [], "o", color=_W_ACCENT, markersize=20,
                          markerfacecolor="none", markeredgewidth=0.7,
                          alpha=0.25)[0]
    ring_inner = ax.plot([], [], "o", color=_W_RED, markersize=12,
                          markerfacecolor="none", markeredgewidth=2.0)[0]
    marker     = ax.plot([], [], "+", color=_W_RED,
                          markersize=20, markeredgewidth=2.2)[0]

    # ── Panel derecho: info ───────────────────────────────────────────────
    INFO_LEFT = 0.625
    ax_info = fig.add_axes([INFO_LEFT, IMG_BOTTOM,
                             1.0 - INFO_LEFT - 0.015, IMG_HEIGHT])
    ax_info.set_facecolor(_W_PANEL)
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.axis("off")
    for sp in ax_info.spines.values():
        sp.set_edgecolor(_W_BORDER); sp.set_linewidth(1.2)

    def _sep(y):
        ax_info.plot([0.06, 0.94], [y, y], color=_W_BORDER,
                     linewidth=0.9, transform=ax_info.transData)

    def _chip(y, label):
        ax_info.text(0.5, y, label, ha="center", va="center",
                     fontsize=7.5, color=_W_DIM, fontfamily="monospace",
                     transform=ax_info.transData,
                     bbox=dict(facecolor=_W_BORDER, pad=2.5,
                               boxstyle="round,pad=0.25", linewidth=0))

    _chip(0.915, "  CURSOR  ")
    cur_text = ax_info.text(0.5, 0.81, "—", ha="center", va="center",
                            fontsize=15, color=_W_TEXT,
                            fontfamily="monospace", linespacing=1.6,
                            transform=ax_info.transData)
    _sep(0.715)

    _chip(0.665, "  DEPTH  ")
    dep_text = ax_info.text(0.5, 0.56, "—", ha="center", va="center",
                            fontsize=19, fontweight="bold", color=_W_TEXT,
                            fontfamily="monospace",
                            transform=ax_info.transData)
    _sep(0.465)

    _chip(0.415, "  SELECTED  ")
    sel_text = ax_info.text(0.5, 0.29, "—", ha="center", va="center",
                            fontsize=13, color=_W_DIM,
                            fontfamily="monospace", linespacing=1.7,
                            transform=ax_info.transData)
    _sep(0.155)

    ax_info.text(0.5, 0.08, "hover → depth preview\nclick → confirm",
                 ha="center", va="center", fontsize=7.5, color=_W_DIM,
                 fontfamily="monospace", linespacing=1.5,
                 transform=ax_info.transData)

    # Punto azul decorativo (esquina inferior derecha del panel)
    ax_info.plot(0.88, 0.04, "o", color=_W_ACCENT, markersize=7,
                 transform=ax_info.transData, clip_on=False)

    # ── Callbacks ────────────────────────────────────────────────────────
    def _on_move(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        u = max(0, min(255, int(round(event.xdata))))
        v = max(0, min(255, int(round(event.ydata))))
        hline.set_ydata([v, v])
        vline.set_xdata([u, u])
        cur_text.set_text(f"u = {u:3d}\nv = {v:3d}")
        d_mm = int(depth_256[v, u])
        if 0 < d_mm < DEPTH_MAX_VALID_MM:
            dep_text.set_text(f"{d_mm} mm")
            dep_text.set_color(_W_GREEN)
        else:
            dep_text.set_text("INVALID")
            dep_text.set_color(_W_RED)
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        u = max(0, min(255, int(round(event.xdata))))
        v = max(0, min(255, int(round(event.ydata))))

        d_mm = int(depth_256[v, u])
        if 0 < d_mm < DEPTH_MAX_VALID_MM:
            d_str = f"{d_mm} mm"
            col   = _W_RED
        else:
            nb    = find_valid_depth(depth_256, u, v, DEPTH_SEARCH_RADIUS)
            d_str = f"~{nb[2]} mm *" if nb else "NO DEPTH"
            col   = _W_AMBER

        marker.set_data([u], [v]);     marker.set_color(col)
        ring_inner.set_data([u], [v]); ring_inner.set_color(col)
        ring_outer.set_data([u], [v]); ring_outer.set_color(col)
        sel_text.set_text(f"({u},  {v})\n{d_str}")
        sel_text.set_color(_W_ACCENT)
        dep_text.set_text(d_str)
        dep_text.set_color(col)
        fig.canvas.draw_idle()

        clicked.clear()
        clicked.append((u, v))

        timer = fig.canvas.new_timer(interval=1000)
        timer.single_shot = True
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)
    fig.canvas.mpl_connect("button_press_event",  _on_click)
    plt.show()

    return clicked[-1] if clicked else None


# ── Contact anchor lifecycle manager ─────────────────────────────────────────

class ContactAnchor:
    """
    Tracks the contact anchor through the grasp sequence.

    Pre-grasp:  anchor expressed in the current EE frame each step
                  p_ee_i = inv(T_ee_to_base_i) @ p_base_h
    Post-grasp: anchor frozen at the EE-frame value at first grasp detection
                  p_ee_frozen  (object moves with the gripper)
    """

    def __init__(
        self,
        p_base_h: np.ndarray,
        contact_loc: np.ndarray,
        contact_scale: np.ndarray,
        close_threshold: float = 0.02,
        freeze_on_close: bool = True,
    ):
        self.p_base_h = p_base_h.astype(np.float64)           # (4,)
        self.contact_loc   = contact_loc.astype(np.float32)   # (3,)
        self.contact_scale = contact_scale.astype(np.float32) # (3,)
        self.close_threshold = close_threshold
        self.freeze_on_close = bool(freeze_on_close)
        self._frozen = False
        self._p_ee_frozen_norm = None  # (3,) float32, set once

    @classmethod
    def from_env(
        cls,
        env,
        T_cam_to_ee: np.ndarray,
        contact_loc: np.ndarray,
        contact_scale: np.ndarray,
        close_threshold: float = 0.02,
        freeze_on_close: bool = True,
    ) -> "ContactAnchor":
        """
        Interactive setup: shows the wrist image, user clicks the contact pixel,
        reads depth, backprojects and transforms to the robot base frame.
        Returns a ContactAnchor ready for rollout.
        """
        print("\n[CONTACT] Waiting for depth and EE pose to be available...")
        t0 = time.time()
        while not env.node.is_contact_ready():
            if time.time() - t0 > 15.0:
                raise RuntimeError("Timeout waiting for depth/EE pose topics")
            time.sleep(0.1)

        rgb_256 = env.node.get_cam0()
        rgb_256 = cv2.resize(rgb_256, (256, 256), interpolation=cv2.INTER_AREA)
        depth_256 = env.get_depth_256()

        print("[CONTACT] Select the contact point in the window that will open...")
        uv = pick_contact_pixel(rgb_256, depth_256)

        if uv is None:
            raise RuntimeError("No contact point selected — window closed without a click")

        u, v = uv
        depth_mm = int(depth_256[v, u])
        used_u, used_v = u, v

        if depth_mm <= 0 or depth_mm >= DEPTH_MAX_VALID_MM:
            nearest = find_valid_depth(depth_256, u, v, DEPTH_SEARCH_RADIUS)
            if nearest is None:
                raise RuntimeError(
                    f"Depth at ({u},{v}) is invalid ({depth_mm} mm) and no valid "
                    f"neighbor found within radius {DEPTH_SEARCH_RADIUS}"
                )
            used_u, used_v, depth_mm = nearest
            print(f"[CONTACT] Fallback depth pixel: ({used_u},{used_v}) = {depth_mm} mm")

        depth_m = depth_mm / 1000.0
        p_cam_h = backproject(used_u, used_v, depth_m)

        # p_cam → p_ee (via hand-eye)
        p_ee_h = T_cam_to_ee @ p_cam_h

        # p_ee → p_base (via current EE pose)
        T_ee_to_base = env.get_ee_T()
        p_base_h = T_ee_to_base @ p_ee_h

        print(f"[CONTACT] Clicked pixel  : ({u}, {v})")
        print(f"[CONTACT] Used depth px  : ({used_u}, {used_v})  depth={depth_mm} mm")
        print(f"[CONTACT] p_cam          : {np.round(p_cam_h[:3], 4)} m")
        print(f"[CONTACT] p_ee           : {np.round(p_ee_h[:3], 4)} m")
        print(f"[CONTACT] p_base         : {np.round(p_base_h[:3], 4)} m")

        return cls(
            p_base_h,
            contact_loc,
            contact_scale,
            close_threshold=close_threshold,
            freeze_on_close=freeze_on_close,
        )

    def get_tensor(
        self,
        T_ee_to_base: np.ndarray,
        measured_gripper: float,
    ) -> torch.Tensor:
        """
        Returns a (1, 3) CUDA tensor of the normalized contact anchor for this step.

        Freezes only when the measured gripper confirms physical contact.
        This matches the converter (detect_contact_frame looks for the gripper
        physically stabilising, not just the close command being issued).
        """
        # PLACE mode: do NOT freeze the anchor. The target is world-fixed in base,
        # exactly like convert_to_robobuf_place_v2.py: every step uses
        # p_ee_i = inv(T_ee_to_base_i) @ p_base_target.
        if not self.freeze_on_close:
            T_base_to_ee = np.linalg.inv(T_ee_to_base)
            p_ee = (T_base_to_ee @ self.p_base_h)[:3].astype(np.float32)
            p_ee_norm_raw = (p_ee - self.contact_loc) / self.contact_scale
            p_ee_norm = np.clip(p_ee_norm_raw, -1, 1)
            print(
                f"[PLACE_DBG] "
                f"p_ee_raw={np.round(p_ee, 4)} "
                f"norm_raw={np.round(p_ee_norm_raw, 3)} "
                f"norm_clip={np.round(p_ee_norm, 3)}"
            )
            return torch.from_numpy(p_ee_norm).float()[None].cuda()  # (1, 3)

        # CONTACT / GRASP mode: freeze once physical close is confirmed.
        if not self._frozen:
            phys_close = measured_gripper < self.close_threshold

            if phys_close:
                T_base_to_ee = np.linalg.inv(T_ee_to_base)
                p_ee = (T_base_to_ee @ self.p_base_h)[:3].astype(np.float32)
                self._p_ee_frozen_norm = np.clip(
                    (p_ee - self.contact_loc) / self.contact_scale, -1, 1
                )
                self._frozen = True
                print(
                    f"[CONTACT] Anchor frozen (physical)  "
                    f"p_ee={np.round(p_ee, 4)}"
                )

        if self._frozen:
            p_ee_norm = self._p_ee_frozen_norm
        else:
            T_base_to_ee = np.linalg.inv(T_ee_to_base)
            p_ee = (T_base_to_ee @ self.p_base_h)[:3].astype(np.float32)
            p_ee_norm_raw = (p_ee - self.contact_loc) / self.contact_scale
            p_ee_norm = np.clip(p_ee_norm_raw, -1, 1)
            print(
                f"[CONTACT_DBG] "
                f"p_ee_raw={np.round(p_ee, 4)} "
                f"norm_raw={np.round(p_ee_norm_raw, 3)} "
                f"norm_clip={np.round(p_ee_norm, 3)}"
            )

        return torch.from_numpy(p_ee_norm).float()[None].cuda()  # (1, 3)


# ── Policy ────────────────────────────────────────────────────────────────────

class Policy:
    def __init__(self, agent_path: str, model_name: str, args):
        self.args = args

        with open(Path(agent_path, "agent_config.yaml"), encoding="utf-8") as f:
            agent_config = yaml.safe_load(f)
        with open(Path(agent_path, "obs_config.yaml"), encoding="utf-8") as f:
            obs_config = yaml.safe_load(f)

        ac_norm_path = (
            Path(args.ac_norm_path).expanduser()
            if args.ac_norm_path
            else Path(agent_path, "ac_norm.json")
        )
        with open(ac_norm_path, encoding="utf-8") as f:
            ac_norm = json.load(f)
        self.loc   = np.array(ac_norm["loc"],   dtype=np.float32)
        self.scale = np.array(ac_norm["scale"], dtype=np.float32)

        # Contact normalization stats (required for contact policy)
        contact_norm_path = (
            Path(args.contact_norm_path).expanduser()
            if args.contact_norm_path
            else Path(agent_path, "contact_norm.json")
        )
        if not contact_norm_path.exists():
            raise FileNotFoundError(
                f"contact_norm.json not found: {contact_norm_path}. "
                "Pass --contact_norm_path explicitly."
            )
        with open(contact_norm_path, encoding="utf-8") as f:
            cn = json.load(f)
        print(f"[INFO] contact_norm from: {contact_norm_path}")
        self.contact_loc   = np.array(cn["loc"],   dtype=np.float32)
        self.contact_scale = np.array(cn["scale"], dtype=np.float32)

        # Build model
        agent = hydra.utils.instantiate(agent_config)
        with torch.serialization.safe_globals(["omegaconf.listconfig.ListConfig"]):
            save_dict = torch.load(
                Path(agent_path, model_name), map_location="cpu", weights_only=False
            )
        agent.load_state_dict(save_dict["model"])
        self.agent = torch.compile(agent.eval().cuda().get_actions)

        self.transform    = hydra.utils.instantiate(obs_config["transform"])
        self.img_keys     = obs_config["imgs"]
        self.pred_horizon = args.pred_horizon
        self.img_chunk    = int(agent_config.get("imgs_per_cam", 1))

        print(f"[INFO] Checkpoint   : {agent_path}/{model_name}")
        print(f"[INFO] Step         : {save_dict.get('global_step', 'unknown')}")
        print(f"[INFO] ac_norm from : {ac_norm_path}")
        print(f"[INFO] loc          : {np.round(self.loc, 4)}")
        print(f"[INFO] scale        : {np.round(self.scale, 4)}")
        print(f"[INFO] gripper  loc[7]={self.loc[7]:.4f}  scale[7]={self.scale[7]:.4f}"
              f"  (expect loc≈0.0425, scale≈0.0375)")
        print(f"[INFO] contact_loc  : {np.round(self.contact_loc, 4)}")
        print(f"[INFO] contact_scale: {np.round(self.contact_scale, 4)}")
        print(f"[INFO] img_keys     : {self.img_keys}")
        print(f"[INFO] img_chunk    : {self.img_chunk}")

        self.reset()

    def reset(self):
        self.last_ac = None
        self.img_history = {k: deque(maxlen=self.img_chunk) for k in self.img_keys}

    def _proc_images(self, img_dict, size=(256, 256)):
        for k in self.img_keys:
            self.img_history[k].append(img_dict[k].copy())

        torch_imgs = {}
        for i, k in enumerate(self.img_keys):
            hist = list(self.img_history[k])
            while len(hist) < self.img_chunk:
                hist.insert(0, hist[0])

            frames = []
            for frame in hist:
                bgr = cv2.resize(frame[:, :, :3], size, interpolation=cv2.INTER_AREA)
                rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0
                frames.append(rgb)

            stacked = torch.stack(frames, dim=0)
            if self.transform is not None:
                stacked = self.transform(stacked)
            torch_imgs[f"cam{i}"] = stacked[None].cuda()

        return torch_imgs

    def _infer(self, obs: dict, contact_point: torch.Tensor) -> np.ndarray:
        """Returns (pred_horizon, ac_dim) normalized actions."""
        imgs  = self._proc_images(obs["images"])
        state = torch.from_numpy(obs["qpos"]).float()[None].cuda()
        with torch.no_grad():
            ac = self.agent(imgs, state, contact_point=contact_point)
            ac = ac[0].detach().cpu().numpy().astype(np.float32)
        return ac[: self.pred_horizon]

    def forward(
        self,
        obs: dict,
        pred_norm: np.ndarray,
        measured_gripper: float,
    ) -> np.ndarray:
        last_ac = self.last_ac if self.last_ac is not None else pred_norm
        self.last_ac = self.args.gamma * pred_norm + (1.0 - self.args.gamma) * last_ac

        target = self.last_ac * self.scale + self.loc
        target = np.clip(target, FR3_JOINT_LIMITS_LOW, FR3_JOINT_LIMITS_HIGH)

        current = obs["qpos"]
        delta   = target - current
        delta[:7] = np.clip(delta[:7], -self.args.dq_limit, self.args.dq_limit)

        # Amplify arm delta once gripper confirms physically closed
        if measured_gripper < self.args.lift_trigger:
            delta[:7] = np.clip(
                delta[:7] * self.args.lift_scale,
                -self.args.dq_limit,
                self.args.dq_limit,
            )

        delta[7:] = np.clip(delta[7:], -self.args.dg_limit, self.args.dg_limit)
        return current + delta


# ── Auto-lift helper ─────────────────────────────────────────────────────────

def go_to_home(env, q_home, hz=5.0, dq_max=0.05, q_thresh=0.05, timeout=30.0):
    """Move arm to q_home via impedance control, keeping gripper closed."""
    q_home  = np.clip(np.asarray(q_home, dtype=np.float32),
                      FR3_JOINT_LIMITS_LOW[:7], FR3_JOINT_LIMITS_HIGH[:7])
    period  = 1.0 / hz
    t0      = time.time()
    print(f"\n[AUTO_LIFT] Moving to home position: {np.round(q_home, 4).tolist()}")
    while True:
        current_q = env.node.get_q()
        err       = np.abs(current_q - q_home)
        if err.max() < q_thresh:
            print(f"[AUTO_LIFT] Reached home (max_err={err.max():.4f} rad)")
            break
        if time.time() - t0 > timeout:
            print(f"[AUTO_LIFT] Timeout ({timeout:.0f}s) — stopping")
            break
        delta = np.clip(q_home - current_q, -dq_max, dq_max)
        q_cmd = np.clip(current_q + delta,
                        FR3_JOINT_LIMITS_LOW[:7], FR3_JOINT_LIMITS_HIGH[:7])
        try:
            env.node.publish_impedance_target(q_cmd)
        except Exception:
            print("[AUTO_LIFT] Publisher context lost — stopping go_to_home cleanly.")
            return
        time.sleep(period)
    try:
        env.node.publish_impedance_target(q_home)
    except Exception:
        pass
    time.sleep(0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument(
        "--task_mode",
        choices=["contact", "place"],
        default="contact",
        help="contact/grasp: freeze anchor when gripper physically closes. "
             "place: keep anchor world-fixed and wait for gripper opening.",
    )

    _default_hand_eye = str(Path(__file__).parent.parent / "converters" / "my_data" / "hand_eye_result.yaml")
    parser.add_argument("--hand_eye", default=_default_hand_eye,
                        help="Path to hand-eye YAML with T_cam_to_ee_4x4")

    parser.add_argument("--T",             default=300,  type=int)
    parser.add_argument("--num_rollouts",  default=1,    type=int)
    parser.add_argument("--pred_horizon",  default=8,    type=int)
    parser.add_argument("--gamma",         default=0.85, type=float)
    parser.add_argument("--action_idx",    default=0,    type=int,
                        help="Which step of the ac_chunk to execute (0=first). "
                             "Try 2-4 if robot stalls post-grasp.")
    parser.add_argument("--lift_scale",    default=1.0,  type=float,
                        help="Multiply arm delta by this when gripper confirms closed.")
    parser.add_argument("--lift_trigger",  default=0.02, type=float,
                        help="Measured gripper width (m) that activates lift_scale "
                             "and freezes the contact anchor.")

    parser.add_argument("--hz",            default=10.0, type=float)
    parser.add_argument("--dq_limit",      default=0.15, type=float)
    parser.add_argument("--dg_limit",      default=0.02, type=float)
    parser.add_argument("--settle_time",   default=0.05, type=float)

    parser.add_argument("--joint_topic",       default="/joint_states")
    parser.add_argument("--image_topic_1",     default="/camera/camera_wrist/color/image_raw",
                        help="cam0 — wrist camera")
    parser.add_argument("--image_topic_2",     default="/camera/camera_ext/color/image_raw",
                        help="cam1 — external camera")
    parser.add_argument("--depth_topic",       default="/camera/camera_wrist/aligned_depth_to_color/image_raw",)
    parser.add_argument("--ee_pose_topic",     default="/franka_robot_state_broadcaster/current_pose")
    parser.add_argument("--impedance_topic",   default="/gello/joint_states")

    parser.add_argument("--open_width",            default=0.08,  type=float)
    parser.add_argument("--close_width",           default=0.005, type=float)
    parser.add_argument("--gripper_speed",         default=0.1,   type=float)
    parser.add_argument("--gripper_force",         default=100.0, type=float)
    parser.add_argument("--gripper_epsilon_inner", default=0.005,  type=float)
    parser.add_argument("--gripper_epsilon_outer", default=0.03,  type=float)

    parser.add_argument("--ac_norm_path", default=None,
                        help="Override path to ac_norm.json.")
    parser.add_argument("--contact_norm_path", default=None,
                        help="Override path to contact_norm.json.")
    parser.add_argument("--dump_obs",     default=None,
                        help="If set, save per-step observations to this .pkl file.")

    parser.add_argument("--auto_lift", action="store_true",
                        help="Una vez el gripper lleva --grasp_confirm_steps cerrado, "
                             "para la política y sube el brazo a --home_q.")
    parser.add_argument("--grasp_confirm_steps", default=5, type=int,
                        help="Nº de steps con gripper cerrado para activar auto_lift en contact mode.")
    parser.add_argument("--release_confirm_steps", default=None, type=int,
                        help="Nº de steps con gripper abierto para activar auto_lift en place mode. "
                             "Si no se especifica, usa --grasp_confirm_steps.")
    parser.add_argument("--place_open_trigger", default=0.065, type=float,
                        help="Measured gripper width (m) que confirma release/open en place mode.")
    parser.add_argument("--place_reset_threshold", default=0.04, type=float,
                        help="En place mode, resetea el contador de release si el gripper cae por debajo de este umbral.")
    parser.add_argument("--place_discrete_gripper", action="store_true",
                        help="En place mode, ejecuta el gripper como comando discreto: si pred_g supera "
                             "--place_open_cmd_threshold manda OPEN_WIDTH directamente; si baja de "
                             "--place_close_cmd_threshold manda CLOSE_WIDTH. Evita que --dg_limit impida abrir.")
    parser.add_argument("--place_open_cmd_threshold", default=0.06, type=float,
                        help="Umbral sobre pred_g denormalizado para mandar apertura discreta en place mode.")
    parser.add_argument("--place_close_cmd_threshold", default=0.02, type=float,
                        help="Umbral sobre pred_g denormalizado para mantener cierre discreto en place mode.")
    parser.add_argument("--grasp_reset_threshold", default=0.04, type=float,
                        help="El contador de grasp solo se resetea si el gripper "
                             "supera este umbral (>lift_trigger). Evita resets por "
                             "fluctuaciones de epsilon. Default 0.04 m.")
    parser.add_argument("--home_q", nargs=7, type=float, default=None, metavar="Q",
                        help="Posición articular de inicio (7 valores en rad). "
                             "Si no se especifica usa [-0.07,-0.71,0.00,-2.45,0.00,1.74,-2.38].")
    parser.add_argument("--home_hz",     default=5.0,  type=float)
    parser.add_argument("--home_dq_max", default=0.05, type=float)
    parser.add_argument("--q_start", nargs=7, type=float, default=None, metavar="Q",
                        help="Configuración articular (7 valores en rad) a la que el "
                             "robot se mueve antes de abrir la GUI de contacto.")
    parser.add_argument("--reset_after_click", action="store_true",
                        help="Llama env.reset() tras la GUI de selección. "
                             "Desactivado por defecto en place para no soltar el objeto.")

    args = parser.parse_args()

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = os.path.basename(args.checkpoint)

    T_cam_to_ee = load_T_cam_to_ee(args.hand_eye)
    print(f"[INFO] T_cam_to_ee loaded from {args.hand_eye}")
    print(f"[INFO] task_mode   : {args.task_mode}")
    if args.task_mode == "place" and args.auto_lift:
        print(
            f"[INFO] PLACE auto_lift: will stop after open gripper is confirmed "
            f"for {args.release_confirm_steps or args.grasp_confirm_steps} steps "
            f"(open_trigger={args.place_open_trigger:.3f} m)"
        )

    policy = Policy(agent_path, model_name, args)

    env = make_fr3_env_2cam_contact(
        init_node=True,
        hz=args.hz,
        dq_limit=args.dq_limit,
        dg_limit=args.dg_limit,
        settle_time=args.settle_time,
        joint_topic=args.joint_topic,
        image_topic_1=args.image_topic_1,
        image_topic_2=args.image_topic_2,
        impedance_topic=args.impedance_topic,
        depth_topic=args.depth_topic,
        ee_pose_topic=args.ee_pose_topic,
        open_width=args.open_width,
        close_width=args.close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_epsilon_inner=args.gripper_epsilon_inner,
        gripper_epsilon_outer=args.gripper_epsilon_outer,
    )

    for rollout_num in range(args.num_rollouts):
        # ── Wait for user confirmation ────────────────────────────────────────
        last_input = None
        while last_input != "y":
            if last_input == "r":
                env.reset()
            last_input = input(
                f"\nRollout {rollout_num + 1}/{args.num_rollouts} — continue? "
                "(y / r=reset gripper): "
            ).strip().lower()

        obs = env.reset()

        # ── Move to start position before contact selection ───────────────────
        q_start = (np.array(args.q_start, dtype=np.float32)
                   if args.q_start is not None
                   else env.node.get_q().copy())
        q_home  = np.array(args.home_q if args.home_q is not None else q_start,
                           dtype=np.float32)
        print(f"\n[ROLLOUT {rollout_num}] Moving to q_start before contact GUI...")
        go_to_home(env, q_start, hz=args.home_hz, dq_max=args.home_dq_max)

        # ── Contact point selection ───────────────────────────────────────────
        print(f"\n[ROLLOUT {rollout_num}] Setting up contact anchor...")
        contact = ContactAnchor.from_env(
            env,
            T_cam_to_ee=T_cam_to_ee,
            contact_loc=policy.contact_loc,
            contact_scale=policy.contact_scale,
            close_threshold=args.lift_trigger,
            freeze_on_close=(args.task_mode == "contact"),
        )

        # For place: do NOT reset after click — it would drop the held object.
        if args.reset_after_click:
            obs = env.reset()
        policy.reset()
        dump_steps = []

        print(f"[ROLLOUT {rollout_num}] Start q: {np.round(q_start, 4).tolist()}")
        print(f"[ROLLOUT {rollout_num}] Starting — {args.T} steps")

        grasp_steps    = 0     # contact mode: consecutive steps with gripper confirmed closed
        release_steps  = 0     # place mode: consecutive steps with gripper confirmed open

        for t in range(args.T):
            measured_gripper = float(env.node.get_gripper())
            T_ee_to_base     = env.get_ee_T()

            # Contact anchor for this step (pre-grasp dynamic / post-grasp frozen).
            # Freeze only on physical close — matches detect_contact_frame in the converter.
            contact_tensor = contact.get_tensor(
                T_ee_to_base,
                measured_gripper,
            )
            # Infer and select action
            preds     = policy._infer(obs.observation, contact_tensor)
            pred_norm = preds[min(args.action_idx, len(preds) - 1)]
            action    = policy.forward(obs.observation, pred_norm, measured_gripper)

            # Logging
            pred_g_denorm = pred_norm[7] * policy.scale[7] + policy.loc[7]

            # PLACE: the gripper action should behave like a discrete release command.
            # The environment only sends OPEN when g_target > env.open_trigger (~0.075).
            # If we leave the policy output rate-limited by --dg_limit, an open prediction
            # such as pred_g=0.08 may become action[7]=0.024 and never trigger OPEN.
            gripper_exec_mode = "policy"
            if args.task_mode == "place" and args.place_discrete_gripper:
                if pred_g_denorm >= args.place_open_cmd_threshold:
                    action[7] = args.open_width
                    gripper_exec_mode = "OPEN"
                elif pred_g_denorm <= args.place_close_cmd_threshold:
                    action[7] = args.close_width
                    gripper_exec_mode = "CLOSE"
                else:
                    gripper_exec_mode = "HOLD"

            current_q = obs.observation["qpos"][:7]
            delta = action[:7] - current_q
            p_ee_norm = contact_tensor[0].cpu().numpy()
            print(
                f"[STEP {t:04d}] "
                f"current={np.round(current_q, 4)}  "
                f"target={np.round(action[:7], 4)}  "
                f"delta={np.round(delta, 4)}  "
                f"|delta|={np.abs(delta).max():.4f}  "
                f"gripper={action[7]:.4f}  "
                f"pred_g={pred_g_denorm:.4f}  "
                f"g_mode={gripper_exec_mode}  "
                f"measured={measured_gripper:.4f}  "
                f"contact_norm={np.round(p_ee_norm, 3)}"
            )

            if args.dump_obs is not None:
                imgs = obs.observation["images"]
                dump_steps.append({
                    "step": t,
                    "raw_state": obs.observation["qpos"].copy(),
                    "pred_action_norm": pred_norm.copy(),
                    "pred_gripper_denorm": float(pred_g_denorm),
                    "measured_gripper": measured_gripper,
                    "contact_norm": p_ee_norm.copy(),
                    "anchor_frozen": contact._frozen,
                    "cam0_jpg": cv2.imencode(".jpg", imgs["cam0"][:, :, ::-1])[1].tobytes(),
                    "cam1_jpg": cv2.imencode(".jpg", imgs["cam1"][:, :, ::-1])[1].tobytes(),
                })

            obs = env.step(action)

            # ── Auto-lift / auto-home ──────────────────────────────────────
            if args.auto_lift:
                if args.task_mode == "contact":
                    # Grasp/contact: stop policy after the gripper has physically closed.
                    if measured_gripper < args.lift_trigger:
                        grasp_steps += 1
                    elif measured_gripper > args.grasp_reset_threshold:
                        # Only reset if gripper is clearly open — ignores epsilon fluctuations
                        grasp_steps = 0

                    if grasp_steps >= args.grasp_confirm_steps:
                        print(f"\n[AUTO_LIFT] Gripper cerrado {grasp_steps} steps → parando política")
                        q_home = (np.array(args.home_q, dtype=np.float32)
                                  if args.home_q is not None
                                  else q_start)
                        go_to_home(env, q_home, hz=args.home_hz, dq_max=args.home_dq_max)
                        break

                else:
                    # Place: stop policy after the gripper has physically opened/released.
                    release_confirm_steps = (
                        args.release_confirm_steps
                        if args.release_confirm_steps is not None
                        else args.grasp_confirm_steps
                    )
                    if measured_gripper > args.place_open_trigger:
                        release_steps += 1
                    elif measured_gripper < args.place_reset_threshold:
                        release_steps = 0

                    if release_steps >= release_confirm_steps:
                        print(f"\n[AUTO_PLACE] Gripper abierto {release_steps} steps → parando política")
                        q_home = (np.array(args.home_q, dtype=np.float32)
                                  if args.home_q is not None
                                  else q_start)
                        go_to_home(env, q_home, hz=args.home_hz, dq_max=args.home_dq_max)
                        break

        if args.dump_obs is not None:
            dump_path = Path(args.dump_obs)
            with open(dump_path, "wb") as f:
                pickle.dump(dump_steps, f)
            print(f"[INFO] Saved {len(dump_steps)}-step dump to {dump_path}")

    env.shutdown()


if __name__ == "__main__":
    main()