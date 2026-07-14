#!/usr/bin/env python3
"""
FR3 eval script — place/contact-conditioned diffusion policy, two cameras.

Workflow per rollout:
  1. User clicks the contact pixel on the live 256×256 wrist image.
  2. Depth at that pixel is read and back-projected to the robot base frame
     using the D435i intrinsics (scaled to 256×256) and hand-eye calibration —
     identical to the training converter.
  3. During the rollout each step:
       place target is fixed in base/world and re-expressed in the current EE frame every step
       when the policy predicts release, a real Franka gripper Move action opens to 0.08 m

Usage:
  python eval_franka_2cam_contact.py path/to/checkpoint.ckpt \\
      --hand_eye my_data/hand_eye_result.yaml
"""

import argparse
import json
import os
import pickle
import subprocess
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


# ── Place target anchor lifecycle manager ──────────────────────────────────────

class PlaceAnchor:
    """
    Tracks a PLACE target selected in the wrist image.

    Difference vs grasping:
      - The selected point is a fixed target in the robot base frame.
      - At every policy step we express that fixed target in the CURRENT EE frame:
            p_ee_i = inv(T_ee_to_base_i) @ p_base_h
      - We DO NOT freeze when the gripper is already closed. For place rollouts the
        gripper is normally closed from the beginning because the object is already
        held, so freezing on "closed" would make the contact_point constant from
        step 0 and break the target-relative conditioning.
    """

    def __init__(
        self,
        p_base_h: np.ndarray,
        contact_loc: np.ndarray,
        contact_scale: np.ndarray,
        debug: bool = True,
    ):
        self.p_base_h = p_base_h.astype(np.float64)           # (4,)
        self.contact_loc   = contact_loc.astype(np.float32)   # (3,)
        self.contact_scale = contact_scale.astype(np.float32) # (3,)
        self.debug = debug

    @classmethod
    def from_env(
        cls,
        env,
        T_cam_to_ee: np.ndarray,
        contact_loc: np.ndarray,
        contact_scale: np.ndarray,
        debug: bool = True,
    ) -> "PlaceAnchor":
        """
        Interactive setup for place: user clicks the desired placing/contact pixel,
        depth is back-projected, then transformed into the robot base frame.
        """
        print("\n[PLACE] Waiting for depth and EE pose to be available...")
        t0 = time.time()
        while not env.node.is_contact_ready():
            if time.time() - t0 > 15.0:
                raise RuntimeError("Timeout waiting for depth/EE pose topics")
            time.sleep(0.1)

        rgb_256 = env.node.get_cam0()
        rgb_256 = cv2.resize(rgb_256, (256, 256), interpolation=cv2.INTER_AREA)
        depth_256 = env.get_depth_256()

        print("[PLACE] Select the desired place/contact point in the window that will open...")
        uv = pick_contact_pixel(rgb_256, depth_256)

        if uv is None:
            raise RuntimeError("No place point selected — window closed without a click")

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
            print(f"[PLACE] Fallback depth pixel: ({used_u},{used_v}) = {depth_mm} mm")

        depth_m = depth_mm / 1000.0
        p_cam_h = backproject(used_u, used_v, depth_m)

        # p_cam → p_ee (via hand-eye)
        p_ee_h = T_cam_to_ee @ p_cam_h

        # p_ee → p_base (via current EE pose). This is the fixed world/place target.
        T_ee_to_base = env.get_ee_T()
        p_base_h = T_ee_to_base @ p_ee_h

        print(f"[PLACE] Clicked pixel  : ({u}, {v})")
        print(f"[PLACE] Used depth px  : ({used_u}, {used_v})  depth={depth_mm} mm")
        print(f"[PLACE] p_cam          : {np.round(p_cam_h[:3], 4)} m")
        print(f"[PLACE] p_ee at click  : {np.round(p_ee_h[:3], 4)} m")
        print(f"[PLACE] p_base target  : {np.round(p_base_h[:3], 4)} m")

        return cls(p_base_h, contact_loc, contact_scale, debug=debug)

    def get_tensor(self, T_ee_to_base: np.ndarray, measured_gripper: float = None) -> torch.Tensor:
        """
        Returns a (1, 3) CUDA tensor of the normalized place target for this step.
        The gripper measurement is accepted only for API compatibility and is not
        used to freeze the anchor.
        """
        T_base_to_ee = np.linalg.inv(T_ee_to_base)
        p_ee = (T_base_to_ee @ self.p_base_h)[:3].astype(np.float32)
        p_ee_norm_raw = (p_ee - self.contact_loc) / self.contact_scale
        p_ee_norm = np.clip(p_ee_norm_raw, -1, 1)

        if self.debug:
            print(
                f"[PLACE_DBG] "
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
    ) -> np.ndarray:
        last_ac = self.last_ac if self.last_ac is not None else pred_norm
        self.last_ac = self.args.gamma * pred_norm + (1.0 - self.args.gamma) * last_ac

        target = self.last_ac * self.scale + self.loc
        target = np.clip(target, FR3_JOINT_LIMITS_LOW, FR3_JOINT_LIMITS_HIGH)

        current = obs["qpos"]
        delta   = target - current
        delta[:7] = np.clip(delta[:7], -self.args.dq_limit, self.args.dq_limit)
        delta[7:] = np.clip(delta[7:], -self.args.dg_limit, self.args.dg_limit)
        return current + delta


# ── Auto-lift helper ─────────────────────────────────────────────────────────

def _run_joint_loop(env, q_target, period, dq_max, q_thresh, timeout, label, joints=None):
    """
    Move toward q_target via impedance.
    If joints is not None, only those indices move; others stay at current value.
    Returns when all tracked joints are within q_thresh of target, or on timeout.
    """
    q_target = np.clip(np.asarray(q_target, dtype=np.float32),
                       FR3_JOINT_LIMITS_LOW[:7], FR3_JOINT_LIMITS_HIGH[:7])
    t0 = time.time()
    while True:
        current_q = env.node.get_q()
        tracked   = list(joints) if joints is not None else list(range(len(q_target)))
        err       = np.max(np.abs(current_q[tracked] - q_target[tracked]))
        if err < q_thresh:
            print(f"[AUTO_LIFT] {label} done (max_err={err:.4f} rad)")
            break
        if time.time() - t0 > timeout:
            print(f"[AUTO_LIFT] {label} timeout ({timeout:.0f}s) — continuing")
            break
        delta = np.zeros(len(q_target), dtype=np.float32)
        for j in tracked:
            delta[j] = np.clip(q_target[j] - current_q[j], -dq_max, dq_max)
        q_cmd = np.clip(current_q + delta,
                        FR3_JOINT_LIMITS_LOW[:7], FR3_JOINT_LIMITS_HIGH[:7])
        env.node.publish_impedance_target(q_cmd)
        time.sleep(period)


def go_to_home(env, q_home, hz=5.0, dq_max=0.05, q_thresh=0.05, timeout=30.0,
               lift_joints=(1, 3)):
    """
    Move arm to q_home in two phases to avoid collisions:
      Phase 1 — raise arm: move only lift_joints (shoulder+elbow) to their home
                values, so the arm goes up before moving sideways.
      Phase 2 — full home: move all joints to q_home.
    Gripper state is left unchanged throughout.
    """
    q_home = np.clip(np.asarray(q_home, dtype=np.float32),
                     FR3_JOINT_LIMITS_LOW[:7], FR3_JOINT_LIMITS_HIGH[:7])
    period = 1.0 / hz

    # Phase 1: raise
    print(f"\n[AUTO_LIFT] Phase 1 — raising arm (joints {list(lift_joints)} → home values)")
    _run_joint_loop(env, q_home, period, dq_max, q_thresh,
                    timeout=timeout * 0.4, label="raise", joints=lift_joints)

    # Phase 2: full home
    print(f"[AUTO_LIFT] Phase 2 — moving to home: {np.round(q_home, 4).tolist()}")
    _run_joint_loop(env, q_home, period, dq_max, q_thresh,
                    timeout=timeout, label="home")

    env.node.publish_impedance_target(q_home)
    time.sleep(0.5)


# ── Place release helper ─────────────────────────────────────────────────────

class ExplicitPlaceRelease:
    """
    Sends a real Franka gripper Move action when the diffusion policy predicts
    a release. This avoids relying on the 8th action dimension as a small
    incremental command, which can leave the gripper half-open in place rollouts.
    """

    def __init__(
        self,
        enabled: bool,
        topic: str,
        open_width: float,
        speed: float,
        pred_threshold: float,
        confirm_steps: int,
        timeout: float,
        stop_policy: bool,
    ):
        self.enabled = enabled
        self.topic = topic
        self.open_width = float(open_width)
        self.speed = float(speed)
        self.pred_threshold = float(pred_threshold)
        self.confirm_steps = int(confirm_steps)
        self.timeout = float(timeout)
        self.stop_policy = bool(stop_policy)
        self._open_votes = 0
        self.released = False

    def update(self, pred_gripper_denorm: float, action_gripper: float) -> bool:
        """
        Returns True exactly when release has been triggered. Uses the denormalized
        policy prediction, and also accepts the filtered action as a backup signal.
        """
        if not self.enabled or self.released:
            return False

        wants_open = (
            pred_gripper_denorm >= self.pred_threshold
            or action_gripper >= self.pred_threshold
        )
        self._open_votes = self._open_votes + 1 if wants_open else 0

        if self._open_votes >= self.confirm_steps:
            self.send_open_goal()
            self.released = True
            return True
        return False

    def send_open_goal(self):
        goal = f"{{width: {self.open_width:.4f}, speed: {self.speed:.4f}}}"
        cmd = [
            "ros2", "action", "send_goal",
            self.topic,
            "franka_msgs/action/Move",
            goal,
        ]
        print(f"\n[PLACE_RELEASE] OPEN -> {self.topic}  {goal}")
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if res.stdout:
                print(res.stdout.strip())
            if res.returncode != 0:
                print(f"[PLACE_RELEASE] WARNING: ros2 action returned {res.returncode}")
        except subprocess.TimeoutExpired:
            # The CLI can stay alive while waiting for the action result. The goal
            # has usually already been sent, so this is not necessarily a failure.
            print(
                f"[PLACE_RELEASE] Timeout after {self.timeout:.1f}s waiting for result; "
                "the open goal may still have been accepted."
            )
        except FileNotFoundError:
            print(
                "[PLACE_RELEASE] ERROR: 'ros2' command not found. Source your ROS 2 "
                "workspace before running this script, or disable --explicit_release."
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)

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
    parser.add_argument("--lift_trigger",  default=0.04, type=float,
                        help="Measured gripper width (m) used only for the optional legacy "
                             "auto_lift. It does NOT freeze the place anchor.")
    parser.add_argument("--reset_after_click", action="store_true",
                        help="Grasp-style option. For place, leave disabled so the object is not "
                             "disturbed after selecting the place target.")
    parser.add_argument("--contact_debug", action="store_true",
                        help="Print normalized place target at each step.")

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
    parser.add_argument("--gripper_force",         default=20.0, type=float)
    parser.add_argument("--gripper_epsilon_inner", default=0.005,  type=float)
    parser.add_argument("--gripper_epsilon_outer", default=0.03,  type=float)

    parser.add_argument("--explicit_release", action="store_true", default=True,
                        help="For place: when the policy predicts open, send a real "
                             "Franka Move action to --release_width. Enabled by default.")
    parser.add_argument("--no_explicit_release", dest="explicit_release", action="store_false",
                        help="Disable explicit Franka gripper Move release and use env.step only.")
    parser.add_argument("--release_topic",
                        default="/franka_gripper/franka_gripper/move",
                        help="Franka gripper Move action topic used for place release.")
    parser.add_argument("--release_width", default=0.08, type=float,
                        help="Width sent to franka_msgs/action/Move when releasing.")
    parser.add_argument("--release_speed", default=0.1, type=float,
                        help="Speed sent to franka_msgs/action/Move when releasing.")
    parser.add_argument("--release_pred_threshold", default=0.06, type=float,
                        help="Trigger explicit release when denormalized predicted gripper "
                             "or filtered action is above this width.")
    parser.add_argument("--release_confirm_steps", default=2, type=int,
                        help="Consecutive open predictions required before sending release.")
    parser.add_argument("--release_timeout", default=4.0, type=float,
                        help="Seconds to wait for the ros2 action CLI result.")
    parser.add_argument("--release_stop_policy", action="store_true", default=False,
                        help="Stop the rollout immediately after explicit release. Disabled by default.")
    parser.add_argument("--no_release_stop_policy", dest="release_stop_policy", action="store_false",
                        help="Keep executing the policy after explicit release unless --release_auto_lift is enabled.")
    parser.add_argument("--release_auto_lift", action="store_true", default=True,
                        help="After explicit release, stop the policy and move the arm to --home_q "
                             "or to the rollout start q. Enabled by default for place.")
    parser.add_argument("--no_release_auto_lift", dest="release_auto_lift", action="store_false",
                        help="After release, keep executing the place policy instead of going home.")
    parser.add_argument("--release_lift_delay", default=0.25, type=float,
                        help="Seconds to wait after the open command before moving home.")

    parser.add_argument("--ac_norm_path", default=None,
                        help="Override path to ac_norm.json.")
    parser.add_argument("--contact_norm_path", default=None,
                        help="Override path to contact_norm.json.")
    parser.add_argument("--dump_obs",     default=None,
                        help="If set, save per-step observations to this .pkl file.")

    parser.add_argument("--auto_lift", action="store_true",
                        help="Legacy grasp-style auto-lift: once the gripper is closed for "
                             "--grasp_confirm_steps, stop the policy and move to --home_q.")
    parser.add_argument("--grasp_confirm_steps", default=5, type=int,
                        help="Nº de steps con gripper cerrado para activar auto_lift.")
    parser.add_argument("--grasp_reset_threshold", default=0.04, type=float,
                        help="El contador de grasp solo se resetea si el gripper "
                             "supera este umbral (>lift_trigger). Evita resets por "
                             "fluctuaciones de epsilon. Default 0.04 m.")
    parser.add_argument("--home_hz",      default=10.0, type=float)
    parser.add_argument("--home_dq_max",  default=0.12, type=float)
    parser.add_argument("--lift_joints",  nargs="+", type=int, default=[1, 3],
                        help="Joint indices moved first in phase-1 raise. Default: 1 3 (shoulder+elbow).")
    _DEFAULT_Q_START_PLACE = [-0.2722940742969513, -0.4978274703025818, 0.07207965850830078
, -2.3805019855499268
, -0.03881292790174484
, 1.8746013641357422
, -2.4673891067504883
    ]
    parser.add_argument("--q_start", nargs=7, type=float,
                        default=_DEFAULT_Q_START_PLACE, metavar="Q",
                        help="Configuración articular antes de abrir la GUI de selección.")
    parser.add_argument("--home_q", nargs=7, type=float, default=None, metavar="Q",
                        help="Posición articular de home tras la ejecución. "
                             "Si no se especifica, usa --q_start.")

    args = parser.parse_args()


    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = os.path.basename(args.checkpoint)

    T_cam_to_ee = load_T_cam_to_ee(args.hand_eye)
    print(f"[INFO] T_cam_to_ee loaded from {args.hand_eye}")

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
        q_start = np.array(args.q_start, dtype=np.float32)
        q_home  = np.array(args.home_q if args.home_q is not None else args.q_start,
                           dtype=np.float32)
        print(f"\n[ROLLOUT {rollout_num}] Moving to q_start before contact GUI...")
        go_to_home(env, q_start, hz=args.home_hz, dq_max=args.home_dq_max,
                   lift_joints=args.lift_joints)

        # ── Place target selection ────────────────────────────────────────────
        print(f"\n[ROLLOUT {rollout_num}] Setting up place target anchor...")
        contact = PlaceAnchor.from_env(
            env,
            T_cam_to_ee=T_cam_to_ee,
            contact_loc=policy.contact_loc,
            contact_scale=policy.contact_scale,
            debug=args.contact_debug,
        )

        # Important for place: do not reset automatically after selecting the
        # target, because reset routines often open/reset the gripper and can drop
        # or disturb the object. Use --reset_after_click only if your env.reset()
        # is known to be safe for place rollouts.
        if args.reset_after_click:
            obs = env.reset()
        policy.reset()
        dump_steps = []

        print(f"[ROLLOUT {rollout_num}] Start q: {np.round(q_start, 4).tolist()}")
        print(f"[ROLLOUT {rollout_num}] Starting — {args.T} steps")

        releaser = ExplicitPlaceRelease(
            enabled=args.explicit_release,
            topic=args.release_topic,
            open_width=args.release_width,
            speed=args.release_speed,
            pred_threshold=args.release_pred_threshold,
            confirm_steps=args.release_confirm_steps,
            timeout=args.release_timeout,
            stop_policy=args.release_stop_policy,
        )
        grasp_steps    = 0     # consecutive steps with gripper confirmed closed

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
            action    = policy.forward(obs.observation, pred_norm)

            # Logging
            pred_g_denorm = pred_norm[7] * policy.scale[7] + policy.loc[7]

            # For place, a full release should be executed as a real Franka gripper
            # Move action, equivalent to:
            #   ros2 action send_goal /franka_gripper/franka_gripper/move \
            #       franka_msgs/action/Move "{width: 0.08, speed: 0.1}"
            # The policy's gripper value is still logged and can be used as the trigger.
            release_now = releaser.update(
                pred_gripper_denorm=float(pred_g_denorm),
                action_gripper=float(action[7]),
            )
            if releaser.released:
                action[7] = args.release_width
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
                    "anchor_frozen": False,
                    "cam0_jpg": cv2.imencode(".jpg", imgs["cam0"][:, :, ::-1])[1].tobytes(),
                    "cam1_jpg": cv2.imencode(".jpg", imgs["cam1"][:, :, ::-1])[1].tobytes(),
                })

            obs = env.step(action)

            if release_now and args.release_auto_lift:
                # Place behavior: once the object has been released with the real
                # Franka Move action, stop the diffusion policy and take the arm
                # back to the home/start configuration. The gripper remains open.
                if args.release_lift_delay > 0:
                    time.sleep(args.release_lift_delay)
                print("\n[PLACE_RELEASE] Release done → returning to place-ready position")
                go_to_home(env, q_home, hz=args.home_hz, dq_max=args.home_dq_max,
                           lift_joints=args.lift_joints)
                break

            if release_now and releaser.stop_policy:
                print("[PLACE_RELEASE] Release done → stopping rollout")
                break

            # ── Auto-lift: para la política y sube una vez agarrado ───────────
            if args.auto_lift:
                if measured_gripper < args.lift_trigger:
                    grasp_steps += 1
                elif measured_gripper > args.grasp_reset_threshold:
                    # Only reset if gripper is clearly open — ignores epsilon fluctuations
                    grasp_steps = 0

                if grasp_steps >= args.grasp_confirm_steps:
                    print(f"\n[AUTO_LIFT] Gripper cerrado {grasp_steps} steps → parando política")
                    go_to_home(env, q_home, hz=args.home_hz, dq_max=args.home_dq_max,
                               lift_joints=args.lift_joints)
                    break

        if args.dump_obs is not None:
            dump_path = Path(args.dump_obs)
            with open(dump_path, "wb") as f:
                pickle.dump(dump_steps, f)
            print(f"[INFO] Saved {len(dump_steps)}-step dump to {dump_path}")

    env.shutdown()


if __name__ == "__main__":
    main()