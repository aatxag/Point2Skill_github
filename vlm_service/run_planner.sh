#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_planner.sh  — Launch service_planner from Point2Skill
#
# Usage:
#   ./run_planner.sh                           # default topics, localization OFF
#   ./run_planner.sh enable_localization:=True # enable Qwen+SAM centroid
#   ./run_planner.sh topic_prompt:=/my_prompt enable_localization:=True
#
# All supported ROS params:
#   topic_prompt          (default: /topic_prompt)
#   topic_response        (default: /planner_response)
#   topic_rgb_wrist       (default: /camera/camera_wrist/color/image_raw)
#   topic_selected_policy (default: /selected_policy)
#   topic_policy_status   (default: /policy_execution_status)
#   topic_contact_point_3d(default: /contact_point_3d)
#   policy_timeout        (default: 120.0)
#   enable_localization   (default: False)
#   tmp_image_dir         (default: /tmp/planner_locate)
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Source ROS2 + workspace ───────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
if [ -f "$SCRIPT_DIR/install/local_setup.bash" ]; then
    source "$SCRIPT_DIR/install/local_setup.bash"
fi
if [ -f /home/labiiwa/ros2_ws/install/setup.bash ]; then
    source /home/labiiwa/ros2_ws/install/setup.bash
fi

# ── Python: ml venv + Point2Skill on PYTHONPATH ───────────────────────────────
PYTHON=/home/labiiwa/venvs/ml/bin/python3
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# ── Build --ros-args from positional arguments ────────────────────────────────
ROS_PARAMS=""
for arg in "$@"; do
    ROS_PARAMS="$ROS_PARAMS -p $arg"
done

echo "[run_planner] Starting service_planner..."
echo "[run_planner] Extra params: $*"
echo ""

exec "$PYTHON" -m service_planner.planner --ros-args $ROS_PARAMS
