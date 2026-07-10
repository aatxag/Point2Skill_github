#!/home/labiiwa/venvs/ml/bin/python3
"""
service_planner — VLM-based task planner that maps natural language to
trained diffusion policies.

Flow:
  1. Receive natural language command on /topic_prompt
  2. Qwen generates a structured multi-step plan  (pick/place steps)
  3. Each step is mapped to a trained policy name via keywords + Qwen VLM
  4. For PICK steps: Qwen detects bbox → SAM refines mask → centroid published
  5. Steps are published sequentially to /selected_policy
  6. Waits for /policy_execution_status before advancing to the next step
"""

import json
import os
import re
import sys
import time
from copy import deepcopy

import cv2
import numpy as np
import rclpy
import torch
from PIL import Image as ImagePIL
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(os.path.dirname(__file__))
import image2numpy

# ─── VLM model (loaded once at import time) ───────────────────────────────────

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL_ID = os.environ.get(
    "POINT2SKILL_VLM_MODEL", "/home/labiiwa/ros2_ws/models/Qwen2.5-VL-3B-Instruct"
)
print(f"[Planner] Loading Qwen from directory: {_MODEL_ID}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    _MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
    use_safetensors=True,
)
processor = AutoProcessor.from_pretrained(_MODEL_ID, local_files_only=True)
print("[Planner] Qwen is ready to work.")

# ─── SAM model (loaded once at import time) ───────────────────────────────────

_SAM_CHECKPOINT = os.environ.get(
    "POINT2SKILL_SAM_CKPT", "/home/labiiwa/ros2_ws/models/sam_vit_b.pth"
)
print(f"[Planner] Loading SAM from: {_SAM_CHECKPOINT}")
_sam = sam_model_registry["vit_b"](checkpoint=_SAM_CHECKPOINT)
_sam.to(_DEVICE)
mask_generator = SamAutomaticMaskGenerator(_sam)
print("[Planner] SAM is ready to work.")

# ─── Policy registry ─────────────────────────────────────────────────────────

POLICY_DESCRIPTIONS = (
    "Available trained diffusion policies (name: description):\n"
    "  open                        : open the drawer\n"
    "  pick_coffee              : pick coffee from the table\n"
    "  place drawer     : place the object holding into the drawer\n"
)

# Ordered longest-first so the substring scan picks the most specific match.
KNOWN_POLICIES = [
    "open",
    "pick_coffee",
    "place_drawer",
]


# ─── Geometry helpers (same as vlm_grasp) ────────────────────────────────────

def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(round(x1)), W - 1))
    y1 = max(0, min(int(round(y1)), H - 1))
    x2 = max(0, min(int(round(x2)), W - 1))
    y2 = max(0, min(int(round(y2)), H - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def expand_bbox(b, W, H, pad=80):
    x1, y1, x2, y2 = map(int, b)
    return [max(0, x1 - pad), max(0, y1 - pad),
            min(W - 1, x2 + pad), min(H - 1, y2 + pad)]


# ─── JSON helpers ─────────────────────────────────────────────────────────────

def _extract_json(text: str):
    """Return parsed JSON from text (strips markdown fences if present)."""
    m = re.search(r"```(?:\w+)?\s*(.*?)\s*```", text, re.DOTALL)
    payload = (m.group(1) if m else text).strip()
    try:
        return json.loads(payload)
    except Exception:
        return None


def _extract_json_obj(text: str):
    """Extract first JSON object {...} from arbitrary text."""
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


# ─── Policy keyword matcher ───────────────────────────────────────────────────

def _keyword_policy(action: str, obj: str, target: str = "") -> str | None:
    a = (action or "").lower().strip()
    o = (obj    or "").lower().strip()
    t = (target or "").lower().strip()

    # ── open primitive ────────────────────────────────────────────────────────
    if a == "open":
        return "open"

    # ── place primitive ───────────────────────────────────────────────────────
    if a in ("place", "put", "drop", "set", "move"):
        if "drawer" in t or "drawer" in o:
            return "place_drawer"
        return None

    # ── pick primitive ────────────────────────────────────────────────────────
    if a in ("pick", "grab", "grasp"):
        if "coffee" in o:
            return "pick_coffee"
        return None

    return None


# ─── Main node ───────────────────────────────────────────────────────────────

class PlannerNode(Node):
    def __init__(self):
        super().__init__("service_planner")

        qos = QoSProfile(depth=10)


        # Parameters
        self.declare_parameter("topic_prompt",          "/topic_prompt")
        self.declare_parameter("topic_response",        "/planner_response")
        self.declare_parameter("topic_rgb_wrist",       "/camera/camera_wrist/color/image_raw")
        self.declare_parameter("topic_selected_policy", "/selected_policy")
        self.declare_parameter("topic_policy_status",   "/policy_execution_status")
        self.declare_parameter("topic_contact_point_3d", "/contact_point_3d")
        self.declare_parameter("policy_timeout",        120.0)
        self.declare_parameter("headless",              True)
        self.declare_parameter("confirm_plan",          True)
        self.declare_parameter("enable_localization",   False)   # Qwen+SAM centroid
        self.declare_parameter("tmp_image_dir",         "/tmp/planner_locate")

        t_prompt   = self.get_parameter("topic_prompt").value
        t_resp     = self.get_parameter("topic_response").value
        t_rgb      = self.get_parameter("topic_rgb_wrist").value
        t_policy   = self.get_parameter("topic_selected_policy").value
        t_pstatus  = self.get_parameter("topic_policy_status").value
        t_centroid = self.get_parameter("topic_contact_point_3d").value
        self.policy_timeout       = float(self.get_parameter("policy_timeout").value)
        self.headless             = bool(self.get_parameter("headless").value)
        self.confirm_plan         = bool(self.get_parameter("confirm_plan").value)
        self.enable_localization  = bool(self.get_parameter("enable_localization").value)
        self.tmp_image_dir        = self.get_parameter("tmp_image_dir").value
        if self.enable_localization:
            os.makedirs(self.tmp_image_dir, exist_ok=True)

        # State
        self.image_rgb     = None
        self.last_prompt   = ""
        self.policy_status = ""

        # Publishers
        self.pub_response = self.create_publisher(String, t_resp,     qos)
        self.pub_policy   = self.create_publisher(String, t_policy,   qos)
        self.pub_centroid = self.create_publisher(String, t_centroid, qos)

        # Allow cb_policy_status to fire while cb_prompt is blocked in _wait_for_policy
        _reentrant = ReentrantCallbackGroup()

        # Subscribers
        self.create_subscription(Image,  t_rgb,     self.cb_rgb,           qos_profile_sensor_data)
        self.create_subscription(String, t_prompt,  self.cb_prompt,        qos,
                                 callback_group=_reentrant)
        self.create_subscription(String, t_pstatus, self.cb_policy_status, qos,
                                 callback_group=_reentrant)

        self.get_logger().info("[Planner] Ready.")
        self.get_logger().info(f"[Planner] Listening on    : {t_prompt}")
        self.get_logger().info(f"[Planner] Publishing to   : {t_policy}")
        self.get_logger().info(f"[Planner] Status topic    : {t_pstatus}")
        self.get_logger().info(
            f"[Planner] Localization    : {'ENABLED → ' + t_centroid if self.enable_localization else 'DISABLED'}"
        )

    # ─── Callbacks ───────────────────────────────────────────────────────────

    def cb_rgb(self, msg):
        self.image_rgb = deepcopy(image2numpy.image2numpy(msg))

    def cb_policy_status(self, msg: String):
        self.policy_status = msg.data.strip()
        self.get_logger().info(f"[Planner] Policy status: {self.policy_status}")

    def cb_prompt(self, msg: String):
        prompt = msg.data.strip()
        if not prompt or prompt == self.last_prompt:
            return
        self.last_prompt = prompt
        self.get_logger().info(f"[Planner] Command: '{prompt}'")
        self.run_plan()

    # ─── Qwen inference helper ────────────────────────────────────────────────

    def _vlm(self, prompt_text: str, image=None, max_tokens: int = 512) -> str:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return processor.batch_decode(
            out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True
        )[0].strip()

    # ─── Step 1 — VLM task planning ──────────────────────────────────────────

    def _generate_plan(self, command: str, img_pil) -> tuple[list | None, str]:
        """Returns (plan_list, raw_json_str). raw_json_str is always set."""
        prompt = (
            "Generate a step-by-step plan to execute the following command.\n\n"
            "Allowed actions:\n"
            "  open  — open a drawer\n\n"
            "  pick  — pick up an object with the gripper\n"
            "  place — place the held object at a target location\n"
            "Ordering rules (MUST follow):\n"
            "  - If the target is a drawer, ALWAYS open it FIRST, then pick, then place.\n"
            "  - Never pick before opening the container that will receive the object.\n\n"
            "Output format — JSON array only, no explanation:\n"
            '[\n'
            '  {"step": 1, "action": "open",  "target": "<container>"},\n'
            '  {"step": 2, "action": "pick",  "object": "<object_name>", "target": null},\n'
            '  {"step": 3, "action": "place", "object": "<object_name>", "target": "<location>"}\n'
            ']\n\n'
            f"Command: {command}"
        )
        raw = self._vlm(prompt, image=img_pil, max_tokens=512)
        self.get_logger().info(f"[Planner] Raw plan output:\n{raw}")

        plan = _extract_json(raw)
        if isinstance(plan, list) and plan:
            return plan, raw
        if isinstance(plan, dict):
            return [plan], raw
        return None, raw

    # ─── Step 2 — Map each step to a policy ──────────────────────────────────

    def _map_to_policy(self, action: str, obj: str, target: str, img_pil) -> str:
        policy = _keyword_policy(action, obj, target)
        if policy:
            self.get_logger().info(f"[Planner] Keyword match → {policy}")
            return policy

        prompt = (
            f"A robotic arm must execute:\n"
            f"  action : {action}\n"
            f"  object : {obj}\n"
            f"  target : {target or 'N/A'}\n\n"
            f"{POLICY_DESCRIPTIONS}\n\n"
            "Which single policy fits best? Reply with ONLY the policy name."
        )
        raw = self._vlm(prompt, image=img_pil, max_tokens=32)
        raw_clean = raw.strip().lower().replace(" ", "_").replace("-", "_")
        self.get_logger().info(f"[Planner] VLM policy answer: '{raw_clean}'")

        for p in KNOWN_POLICIES:
            if p in raw_clean:
                self.get_logger().info(f"[Planner] VLM match → {p}")
                return p

        fallback = "pick_coffee" if action.lower() in ("pick", "grab", "grasp") else "place_drawer"
        self.get_logger().warning(f"[Planner] No match, fallback → {fallback}")
        return fallback

    # ─── Object centroid: Qwen bbox + SAM mask refinement ────────────────────

    def _locate_object(self, label: str, img_pil) -> dict | None:
        """
        Locate 'label' in img_pil using the same approach as vlm_grasp:
          1. Qwen detects bounding box
          2. SAM generates masks on a crop around the bbox
          3. Best mask is selected by coverage + inside_ratio scoring
          4. Centroid computed from mask (or bbox center as fallback)
        Returns {"u": px, "v": py, "found": True/False, "label": label}
        Saves annotated debug image to tmp_image_dir.
        """
        if img_pil is None:
            self.get_logger().warn("[Locate] No image available.")
            return None

        frame_rgb = np.array(img_pil)
        H, W = frame_rgb.shape[:2]

        # ── 1. Qwen bbox detection ────────────────────────────────────────────
        prompt_text = (
            f"You are given an image of size {W}x{H} pixels.\n"
            f"Detect ONLY the object requested: '{label}'.\n"
            "Return exactly one JSON object in this format:\n"
            '{"label":"<string>", "bbox_2d":[x1,y1,x2,y2], "success":true|false}\n'
            "Coordinates (x1,y1,x2,y2) must be absolute pixel values.\n"
            "Output only the JSON, nothing else."
        )
        raw_det = self._vlm(prompt_text, image=img_pil, max_tokens=128)
        self.get_logger().info(f"[Locate] Qwen raw: {raw_det}")

        det = _extract_json(raw_det)
        if isinstance(det, list) and det:
            det = det[0]
        if not isinstance(det, dict):
            det = _extract_json_obj(raw_det) or {}

        bbox  = det.get("bbox_2d", [0, 0, 0, 0])
        found = bool(det.get("success", False))

        if not found or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            self.get_logger().warn(f"[Locate] Qwen did not detect '{label}'.")
            return {"u": None, "v": None, "found": False, "label": label}

        qwen_bbox = clamp_bbox(*bbox, W, H)
        if qwen_bbox == [0, 0, 0, 0]:
            return {"u": None, "v": None, "found": False, "label": label}

        self.get_logger().info(f"[Locate] Qwen bbox → {qwen_bbox}")

        # ── 2. SAM mask refinement ────────────────────────────────────────────
        pad = 200
        rx1, ry1, rx2, ry2 = expand_bbox(qwen_bbox, W, H, pad=pad)
        crop_rgb  = frame_rgb[ry1:ry2 + 1, rx1:rx2 + 1].copy()
        crop_H, crop_W = crop_rgb.shape[:2]

        # Qwen bbox center in crop coords
        qx = max(0, min(crop_W - 1, int((qwen_bbox[0] + qwen_bbox[2]) / 2) - rx1))
        qy = max(0, min(crop_H - 1, int((qwen_bbox[1] + qwen_bbox[3]) / 2) - ry1))

        # bbox in crop coords
        bx1 = max(0, min(crop_W - 1, int(qwen_bbox[0] - rx1)))
        by1 = max(0, min(crop_H - 1, int(qwen_bbox[1] - ry1)))
        bx2 = max(0, min(crop_W - 1, int(qwen_bbox[2] - rx1)))
        by2 = max(0, min(crop_H - 1, int(qwen_bbox[3] - ry1)))

        bbox_mask = np.zeros((crop_H, crop_W), dtype=bool)
        bbox_mask[by1:by2 + 1, bx1:bx2 + 1] = True
        bbox_area = float(bbox_mask.sum()) or 1.0

        try:
            sam_masks = mask_generator.generate(crop_rgb)
        except torch.cuda.OutOfMemoryError:
            self.get_logger().error("[Locate] SAM CUDA OOM — using Qwen bbox center.")
            torch.cuda.empty_cache()
            sam_masks = []
        except Exception as e:
            self.get_logger().error(f"[Locate] SAM error: {e}")
            sam_masks = []

        MIN_AREA = 300
        MAX_AREA = int(0.70 * crop_H * crop_W)
        best_mask  = None
        best_score = -1e18

        for m in sam_masks:
            seg = m.get("segmentation")
            if seg is None:
                continue
            mask = seg.astype(bool)
            area = int(mask.sum())
            if area < MIN_AREA or area > MAX_AREA:
                continue
            inter = int((mask & bbox_mask).sum())
            if inter == 0:
                continue

            coverage       = inter / bbox_area
            inside_ratio   = inter / float(area)
            contains_center = 1.0 if mask[qy, qx] else 0.0
            touches_border  = bool(
                mask[0, :].any() or mask[-1, :].any() or
                mask[:, 0].any() or mask[:, -1].any()
            )
            border_penalty = 0.35 if touches_border else 0.0
            area_ratio     = area / float(max(crop_H * crop_W, 1))

            score = (
                2.0 * coverage
                + 3.0 * inside_ratio
                + 1.5 * contains_center
                - 1.5 * area_ratio
                - border_penalty
            )
            if score > best_score:
                best_score = score
                best_mask  = mask

        if best_mask is None:
            self.get_logger().warn("[Locate] SAM found no suitable mask — using Qwen bbox center.")
            px_crop, py_crop = qx, qy
        else:
            ys, xs = np.where(best_mask)
            px_crop = int(xs.mean())
            py_crop = int(ys.mean())

        # Back to full-image coords
        px = max(0, min(W - 1, px_crop + rx1))
        py = max(0, min(H - 1, py_crop + ry1))
        self.get_logger().info(f"[Locate] '{label}' centroid → ({px}, {py})")

        # ── 3. Annotate and save debug image ──────────────────────────────────
        try:
            full_mask = None
            if best_mask is not None:
                full_mask = np.zeros((H, W), dtype=bool)
                full_mask[ry1:ry2 + 1, rx1:rx2 + 1] = best_mask.astype(bool)

            annotated = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)

            # Green SAM mask overlay
            if full_mask is not None:
                green = annotated.copy()
                green[full_mask] = [0, 255, 0]
                annotated = cv2.addWeighted(annotated, 0.6, green, 0.4, 0)

            # Qwen bbox
            x1b, y1b, x2b, y2b = qwen_bbox
            cv2.rectangle(annotated, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)

            # Centroid cross + circle + label
            cv2.drawMarker(annotated, (px, py), (0, 0, 255),
                           cv2.MARKER_CROSS, 50, 3)
            cv2.circle(annotated, (px, py), 12, (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"{label} ({px},{py})",
                (px + 16, max(py - 16, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )

            ts = int(time.time())
            out_path = os.path.join(
                self.tmp_image_dir,
                f"locate_{label.replace(' ', '_')}_{ts}.png",
            )
            cv2.imwrite(out_path, annotated)
            self.get_logger().info(f"[Locate] Annotated image → {out_path}")
        except Exception as e:
            self.get_logger().warn(f"[Locate] Annotation failed: {e}")
            out_path = ""

        return {"u": px, "v": py, "found": True, "label": label, "image_path": out_path}

    # ─── Step 3 — Execute plan sequentially ──────────────────────────────────

    def _wait_for_policy(self, policy_name: str) -> bool:
        t0 = time.time()
        self.get_logger().info(f"[Planner] Waiting for policy '{policy_name}'…")
        while rclpy.ok():
            s = self.policy_status
            if f"success:{policy_name}" in s:
                return True
            if any(k in s for k in ("failed:", "error:", "stopped:")):
                self.get_logger().warning(f"[Planner] Execution ended with: {s}")
                return False
            if time.time() - t0 > self.policy_timeout:
                self.get_logger().error(f"[Planner] Timeout after {self.policy_timeout}s")
                return False
            time.sleep(0.1)
        return False

    def _pub_response(self, text: str):
        m = String()
        m.data = text
        self.pub_response.publish(m)
        self.get_logger().info(f"[Planner] → {text}")

    # ─── Plan helpers ────────────────────────────────────────────────────────

    def _map_steps(self, plan: list, img_pil) -> list:
        mapped = []
        for i, step in enumerate(plan, 1):
            action = str(step.get("action", "pick")).strip()
            obj    = str(step.get("object", "") or "").strip()
            target = str(step.get("target") or "").strip()
            policy = self._map_to_policy(action, obj, target, img_pil)
            mapped.append({"step": i, "action": action, "object": obj,
                           "target": target, "policy": policy})
        return mapped

    def _gui_confirm_raw_plan(self, command: str, raw_json: str) -> str | None:
        """
        Open a tkinter window showing the raw VLM plan JSON.
        The user can edit the JSON, then click Accept or Cancel.
        Returns the (possibly edited) JSON string, or None if cancelled.
        Blocks the calling thread until the window is closed.
        """
        import threading
        try:
            import tkinter as tk
            from tkinter import scrolledtext
        except ImportError:
            self.get_logger().warning("[Planner] tkinter not available — skipping GUI.")
            return raw_json

        result = {"value": None}
        done   = threading.Event()

        def run_gui():
            try:
                root = tk.Tk()
                root.title("Plan VLM — Revisión")
                root.configure(bg="#2b2b2b")
                root.attributes("-topmost", True)

                tk.Label(
                    root,
                    text=f"Comando:  '{command}'",
                    font=("Monospace", 11, "bold"),
                    fg="#e0e0e0", bg="#2b2b2b",
                    wraplength=680, justify="left", anchor="w",
                ).pack(fill="x", padx=14, pady=(14, 2))

                tk.Label(
                    root,
                    text="Plan generado por el VLM  (edita si es incorrecto):",
                    font=("Monospace", 9),
                    fg="#aaaaaa", bg="#2b2b2b", anchor="w",
                ).pack(fill="x", padx=14, pady=(0, 4))

                text = scrolledtext.ScrolledText(
                    root,
                    width=80, height=14,
                    font=("Monospace", 10),
                    bg="#1e1e1e", fg="#d4d4d4",
                    insertbackground="white",
                    relief="flat", bd=6,
                )
                text.pack(fill="both", expand=True, padx=14, pady=4)
                text.insert("1.0", raw_json)
                text.focus_set()

                btn_frame = tk.Frame(root, bg="#2b2b2b")
                btn_frame.pack(fill="x", padx=14, pady=12)

                def accept():
                    result["value"] = text.get("1.0", "end-1c").strip()
                    root.destroy()

                def cancel():
                    result["value"] = None
                    root.destroy()

                tk.Button(
                    btn_frame, text="✓  Aceptar", command=accept,
                    font=("sans-serif", 11, "bold"),
                    bg="#388e3c", fg="white", activebackground="#2e7d32",
                    relief="flat", padx=24, pady=8, cursor="hand2",
                ).pack(side="left", padx=(0, 10))

                tk.Button(
                    btn_frame, text="✗  Cancelar", command=cancel,
                    font=("sans-serif", 11),
                    bg="#c62828", fg="white", activebackground="#b71c1c",
                    relief="flat", padx=24, pady=8, cursor="hand2",
                ).pack(side="left")

                root.update_idletasks()
                w, h = 740, 460
                sw = root.winfo_screenwidth()
                sh = root.winfo_screenheight()
                root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
                root.mainloop()
            except Exception as exc:
                self.get_logger().error(f"[Planner] GUI error: {exc} — auto-accepting plan.")
                result["value"] = raw_json
            finally:
                done.set()

        self.get_logger().info(
            f"[Planner] Waiting for GUI plan confirmation…  (DISPLAY={os.environ.get('DISPLAY', 'NOT SET')})"
        )
        threading.Thread(target=run_gui, daemon=True).start()
        done.wait(timeout=300.0)
        if not done.is_set():
            self.get_logger().warning("[Planner] GUI timed out (300s) — auto-accepting plan.")
            result["value"] = raw_json
        return result["value"]

    # ─── Main orchestration ──────────────────────────────────────────────────

    def run_plan(self):
        img_pil = None
        if self.image_rgb is not None:
            img_pil = ImagePIL.fromarray(self.image_rgb, mode="RGB")

        # ── 1. Generate raw plan ──────────────────────────────────────────────
        self._pub_response("VLM is generating plan…")
        plan, raw_json = self._generate_plan(self.last_prompt, img_pil)

        if plan is None:
            self._pub_response(f"Planning failed for: '{self.last_prompt}'")
            return

        # ── 2. GUI confirmation: show raw JSON, let user edit/accept ──────────
        print(f"[DEBUG] confirm_plan={self.confirm_plan!r}  headless={self.headless!r}", flush=True)
        if self.confirm_plan:
            print("[DEBUG] Calling GUI...", flush=True)
            confirmed_json = self._gui_confirm_raw_plan(self.last_prompt, raw_json)
            print(f"[DEBUG] GUI returned: {confirmed_json!r}", flush=True)
            if confirmed_json is None:
                self._pub_response("Plan cancelled by user.")
                return
            edited = _extract_json(confirmed_json)
            if isinstance(edited, list) and edited:
                plan = edited
            elif isinstance(edited, dict):
                plan = [edited]
            # If the edited text can't be parsed, keep the original plan

        # ── 3. Map steps to policies ──────────────────────────────────────────
        mapped = self._map_steps(plan, img_pil)

        # ── 4. Print accepted plan ────────────────────────────────────────────
        lines = []
        for s in mapped:
            suffix = f" → {s['target']}" if s["target"] else ""
            lines.append(
                f"  {s['step']}. {s['action'].upper()} {s['object']}{suffix}"
                f"  [policy: {s['policy']}]"
            )
        plan_text = "Execution plan:\n" + "\n".join(lines)
        self.get_logger().info(f"[Planner]\n{plan_text}")
        self._pub_response(plan_text)

        # ── 5. Execute sequentially ───────────────────────────────────────────
        for s in mapped:
            policy    = s["policy"]
            action    = s["action"].lower()
            obj_label = s["object"]
            step_info = f"Step {s['step']}: {s['action']} '{obj_label}' → {policy}"

            self.get_logger().info(f"[Planner] Executing {step_info}")
            self._pub_response(f"Executing {step_info}")

            # ── Locate centroid before PICK (only when enabled) ───────────────
            if self.enable_localization and action in ("pick", "grab", "grasp"):
                self.get_logger().info(
                    f"[Planner] Locating centroid for '{obj_label}'…"
                )
                loc = self._locate_object(obj_label, img_pil)
                if loc:
                    centroid_msg = String()
                    centroid_msg.data = json.dumps(loc, ensure_ascii=False)
                    self.pub_centroid.publish(centroid_msg)
                    if loc.get("found"):
                        self.get_logger().info(
                            f"[Planner] ✓ Centroid published: u={loc['u']}, v={loc['v']}"
                        )
                    else:
                        self.get_logger().warn(
                            f"[Planner] '{obj_label}' not found in image."
                        )

            # ── Publish policy and wait ────────────────────────────────────────
            self.policy_status = ""
            m = String()
            m.data = policy
            self.pub_policy.publish(m)

            if not self._wait_for_policy(policy):
                self._pub_response(
                    f"Step {s['step']} FAILED ({policy}) — plan aborted."
                )
                return

        self._pub_response("Plan completed successfully.")
        self.get_logger().info("[Planner] All steps done.")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = PlannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
