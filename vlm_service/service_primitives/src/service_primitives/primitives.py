#!/home/labiiwa/venvs/ml/bin/python3

import os
import select
import subprocess
import termios
import threading
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String

# ─── Policy registry ─────────────────────────────────────────────────────────
_CONDA_PYTHON   = "/home/labiiwa/miniconda3/envs/data4robotics_py310/bin/python3"
_EVAL_DIR       = "/home/labiiwa/Point2Skill_github/dit-policy/eval_scripts"
_BC_BASE        = "/home/labiiwa/Point2Skill_github/dit-policy/bc_finetune"
_GRIPPER_TOPIC  = "/franka_gripper/franka_gripper/move"
_GRIPPER_OPEN   = "{width: 0.08, speed: 0.1}"

_SCRIPTS = {
    "contact":        f"{_EVAL_DIR}/eval_franka_2cam_contact.py",
    "place_contact":  f"{_EVAL_DIR}/eval_franka_2cam_place_contact.py",
    "contact_place":  f"{_EVAL_DIR}/eval_franka_2cam_contact_place.py",
}


def _policy(script_key, ckpt, args, q_start=None, home_q=None):
    """Build a policy entry, appending --q_start / --home_q when provided."""
    full_args = list(args)
    if q_start is not None:
        full_args += ["--q_start"] + [str(v) for v in q_start]
    if home_q is not None:
        full_args += ["--home_q"] + [str(v) for v in home_q]
    return {"script": _SCRIPTS[script_key], "ckpt": ckpt, "args": full_args}


# ┌─────────────────┬────────────────────────────────────────────────────────┐
# │ Policy          │ Description                                            │
# ├─────────────────┼────────────────────────────────────────────────────────┤
# │ open            │ Open the drawer                                        │
# │ pick_coffee     │ Pick coffee cup from the table                         │
# │ place_drawer    │ Place the held object into the open drawer             │
# └─────────────────┴────────────────────────────────────────────────────────┘
MODELS = {

# Auto lift barik
#     "open": _policy(
#         "contact",
#         f"{_BC_BASE}/open/wandb_None_franka_2cam_contact_resnet_gn_2026-06-09_14-25-12/open.ckpt",
#         args    = ["--gamma", "1.0", "--T", "500", "--action_idx", "3", "--lift_scale", "4.0"],
#         q_start = [-0.07029999792575836, -1.2699999809265137,  0.16349999606609344,
#                    -2.759500026702881,    0.20419999957084656,  1.79830002784729,
#                    -2.3066000938415527],
#         home_q  =  [-0.3440000116825104, -0.2078000009059906, 0.2944999933242798, -2.658799886703491, 0.16580000519752502, 2.3619000911712646, -2.5455000400543213]
# ,
#     ),

# q_start zaharra: [-0.07029999792575836, -1.2699999809265137,  0.16349999606609344,
#                    -2.759500026702881,    0.20419999957084656,  1.79830002784729,
#                    -2.3066000938415527]


    "open": _policy(
        "contact",
        f"{_BC_BASE}/open/wandb_None_franka_2cam_contact_resnet_gn_2026-06-09_14-25-12/open.ckpt",
        args    = ["--gamma", "1.0", "--T", "500", "--action_idx", "3", "--auto_lift", "--grasp_confirm_steps", "3", "--lift_trigger", "0.04"],
        q_start = [-0.0714000016450882, -1.264799952507019, 0.10849999636411667, -2.9293999671936035, 0.1451999992132187, 2.075200080871582, -2.3701999187469482],
        home_q  =  [-0.3440000116825104, -0.2078000009059906, 0.2944999933242798, -2.658799886703491, 0.16580000519752502, 2.3619000911712646, -2.5455000400543213]

    ),

    "close": _policy(
        "contact",
        f"{_BC_BASE}/close/wandb_None_franka_2cam_contact_resnet_gn_2026-06-12_18-46-27/close.ckpt",
        args    = ["--gamma", "1.0", "--T", "500", "--action_idx", "3", "--auto_lift", "--grasp_confirm_steps", "3", "--lift_trigger", "0.04"],
        q_start = [-0.08370000123977661, -1.4092999696731567, 0.09730000048875809, -2.8638999462127686, 0.14139999449253082, 1.7776000499725342, -2.3919999599456787],
        home_q  =  [-0.29019999504089355, -0.5712000131607056,  0.15060000121593475,
                   -2.533400058746338,    0.11640000343322754,  1.8801000118255615,
                   -2.587100028991699]
,
    ),

    "pick_coffee": _policy(
        "contact",
        f"{_BC_BASE}/pick_coffee/wandb_None_franka_2cam_contact_resnet_gn_2026-06-12_17-35-22/pick_coffee.ckpt",
        #f"{_BC_BASE}/pick_coffee/wandb_None_franka_2cam_contact_resnet_gn_2026-06-10_10-53-01/pick_coffee.ckpt",

        args    = ["--gamma", "1.0", "--T", "500", "--action_idx", "3",
                   "--auto_lift", "--grasp_confirm_steps", "3", "--lift_trigger", "0.04"],
        q_start = [-0.29019999504089355, -0.5712000131607056,  0.15060000121593475,
                   -2.533400058746338,    0.11640000343322754,  1.8801000118255615,
                   -2.587100028991699],
        home_q  = [-0.1436000019311905,  -0.9527000188827515,  0.3264000117778778,
                   -2.8459999561309814,   0.3375000059604645,   1.8614000082015991,
                   -2.4156999588012695],
    ),

    "place_drawer": _policy(
        "contact_place",
        "/home/labiiwa/dit-policy/bc_finetune/place_drawer"
        "/wandb_None_franka_2cam_contact_resnet_gn_2026-06-10_12-51-05/place_drawer.ckpt",
        args = [
            "--task_mode",                 "place",
            "--gamma",                     "1.0",
            "--T",                         "500",
            "--action_idx",                "3",
            "--lift_scale",                "1.0",
            "--auto_lift",
            "--release_confirm_steps",     "5",
            "--place_open_trigger",        "0.065",
            "--place_discrete_gripper",
            "--place_open_cmd_threshold",  "0.06",
            "--place_close_cmd_threshold", "0.02",
            "--ac_norm_path",
            "/home/labiiwa/dit-policy/bc_finetune/place_drawer"
            "/wandb_None_franka_2cam_contact_resnet_gn_2026-06-10_12-51-05/ac_norm.json",
            "--contact_norm_path",
            "/home/labiiwa/dit-policy/bc_finetune/place_drawer"
            "/wandb_None_franka_2cam_contact_resnet_gn_2026-06-10_12-51-05/contact_norm.json",
        ],
        q_start = [-0.21559999883174896, -0.618399977684021,  -0.03420000150799751,
                   -2.454400062561035,    0.06509999930858612,  1.937600016593933,
                   -2.5927000045776367],
        home_q  = [-0.10779999941587448, -0.16369999945163727, 0.07090000063180923, -2.1473000049591064, -0.033399999141693115, 1.9945000410079956, -2.4207000732421875]
,
    ),

}


class PrimitivesNode(Node):
    def __init__(self):
        super().__init__("service_primitives")

        qos = QoSProfile(depth=10)

        self.declare_parameter("topic_selected_policy", "/selected_policy")
        self.declare_parameter("topic_policy_stop",     "/policy_stop")
        self.declare_parameter("topic_policy_status",   "/policy_execution_status")

        topic_sel    = self.get_parameter("topic_selected_policy").value
        topic_stop   = self.get_parameter("topic_policy_stop").value
        topic_status = self.get_parameter("topic_policy_status").value

        self.pub_status = self.create_publisher(String, topic_status, qos)
        self.create_subscription(String, topic_sel,  self.cb_selected_policy, qos)
        self.create_subscription(String, topic_stop, self.cb_stop, qos)

        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._current_policy = ""
        self._skip_stop_status = False   # set True during emergency stop

        self.get_logger().info(
            f"[Primitives] Ready. Listening for policies on '{topic_sel}'"
        )
        self.get_logger().info(
            f"[Primitives] Available policies: {list(MODELS.keys())}"
        )
        self.get_logger().info(
            "[Primitives] Press 's' to stop current policy, open gripper, and advance plan."
        )

        # Start keyboard listener thread
        kb = threading.Thread(target=self._keyboard_listener, daemon=True)
        kb.start()

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _pub_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)
        self.get_logger().info(f"[Primitives] Status → {status}")

    # ─── keyboard listener ────────────────────────────────────────────────────

    def _keyboard_listener(self):
        """Read single keypresses from /dev/tty (works even when stdin is piped)."""
        try:
            fd = os.open('/dev/tty', os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            self.get_logger().warning(f"[Primitives] Keyboard listener disabled: {e}")
            return

        old_attrs = None
        try:
            old_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)   # char-by-char but keeps Ctrl+C → SIGINT
            while rclpy.ok():
                r, _, _ = select.select([fd], [], [], 0.2)
                if r:
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    if ch.lower() == "s" and self._current_policy:
                        policy = self._current_policy
                        self.get_logger().info(
                            f"[Primitives] 's' pressed — stopping '{policy}', "
                            "opening gripper, advancing plan."
                        )
                        threading.Thread(
                            target=self._emergency_stop,
                            args=(policy,),
                            daemon=True,
                        ).start()
        except Exception as e:
            self.get_logger().warning(f"[Primitives] Keyboard listener error: {e}")
        finally:
            if old_attrs is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
                except Exception:
                    pass
            try:
                os.close(fd)
            except Exception:
                pass

    def _emergency_stop(self, policy_name: str):
        """Stop subprocess → open gripper → publish success so planner continues."""
        self._skip_stop_status = True

        # Let the subprocess finish naturally first (go_to_home may be in progress).
        # If it exits on its own within 5 s we skip the SIGTERM entirely.
        with self._proc_lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass  # still running — _stop_current will SIGTERM below

        self._stop_current()   # no-op if already exited; SIGTERM otherwise

        # Open gripper
        try:
            subprocess.run(
                ["ros2", "action", "send_goal",
                 _GRIPPER_TOPIC, "franka_msgs/action/Move", _GRIPPER_OPEN],
                timeout=8.0,
                capture_output=True,
            )
            self.get_logger().info("[Primitives] Gripper opened.")
        except subprocess.TimeoutExpired:
            self.get_logger().warning(
                "[Primitives] Gripper open timed out (goal may still have been sent)."
            )
        except Exception as e:
            self.get_logger().error(f"[Primitives] Gripper open error: {e}")
        finally:
            self._skip_stop_status = False
            self._pub_status(f"success:{policy_name}")

    # ─── callbacks ───────────────────────────────────────────────────────────

    def cb_selected_policy(self, msg: String):
        policy_name = msg.data.strip()
        self.get_logger().info(f"[Primitives] Policy requested: '{policy_name}'")

        if policy_name not in MODELS:
            self.get_logger().error(
                f"[Primitives] Unknown policy '{policy_name}'. "
                f"Available: {list(MODELS.keys())}"
            )
            self._pub_status(f"error:unknown_policy:{policy_name}")
            return

        self._stop_current()
        self._current_policy = policy_name

        entry = MODELS[policy_name]
        cmd = [_CONDA_PYTHON, entry["script"], entry["ckpt"]] + entry["args"]

        self.get_logger().info(f"[Primitives] CMD: {' '.join(cmd)}")
        self._pub_status(f"running:{policy_name}")

        thread = threading.Thread(
            target=self._run_subprocess,
            args=(cmd, policy_name),
            daemon=True,
        )
        thread.start()

    def cb_stop(self, msg: String):
        self.get_logger().info("[Primitives] Stop requested via topic.")
        self._stop_current()
        self._pub_status("stopped:user_request")

    # ─── subprocess management ───────────────────────────────────────────────

    def _run_subprocess(self, cmd: list, policy_name: str):
        try:
            proc = subprocess.Popen(cmd, cwd=_EVAL_DIR, stdin=subprocess.PIPE)
            with self._proc_lock:
                self._proc = proc

            # Pre-fill stdin with 'y' answers for the rollout confirmation prompts.
            try:
                proc.stdin.write(b"y\n" * 20)
                proc.stdin.flush()
                proc.stdin.close()
            except OSError:
                pass

            retcode = proc.wait()

            with self._proc_lock:
                self._proc = None

            if self._skip_stop_status:
                # emergency_stop is handling status — don't publish anything here.
                pass
            elif retcode == 0:
                self._pub_status(f"success:{policy_name}")
            elif retcode in (-15, -9):  # SIGTERM / SIGKILL
                self._pub_status(f"stopped:{policy_name}")
            else:
                self._pub_status(f"failed:{policy_name}:code={retcode}")

        except Exception as exc:
            self.get_logger().error(f"[Primitives] Subprocess error: {exc}")
            self._pub_status(f"error:{policy_name}:{exc}")
            with self._proc_lock:
                self._proc = None

    def _stop_current(self):
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.get_logger().info("[Primitives] Terminating running policy...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    def destroy_node(self):
        self._stop_current()
        super().destroy_node()


# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = PrimitivesNode()
    try:
        rclpy.spin(node)
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
