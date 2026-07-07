# POLITIKAK EXEKUTATU AHAL IZATEKO

1. Terminala

conda activate data4robotics_py310
source ~/franka_ros2_ws/install/setup.bash
 ros2 launch franka_bringup franka.launch.py     robot_type:=fr3     robot_ip:=172.17.6.130     load_gripper:=false


2. Terminala

source ~/franka_ros2_ws/install/setup.bash
ros2 launch franka_gripper gripper.launch.py robot_ip:=172.17.6.130 namespace:=franka_gripper

4. Terminala

ros2 launch franka_demo_recorder record_demo_two_cam.launch.py task:=pick_experiments  serial_wrist:="'141122079579'" usb_device:=/dev/video6

ros2 launch franka_demo_recorder record_demo_two_cam_place.launch.py task:=close  serial_wrist:="'141122079579'" usb_device:=/dev/video6


5. Terminala

ros2 run franka_demo_recorder episode_manager   --ros-args --params-file src/franka_demo_recorder/config/recorder_params.yaml

ros2 run franka_demo_recorder episode_manager_place   --ros-args --params-file src/franka_demo_recorder/config/recorder_params.yaml


# Lehen posizioak ikusi nahi badia

 1948  cd dit-policy
 1949  cd eval_scripts/
 1950  conda activate data4robotics_py310
 1951  python3 eval_franka_2cam_contact_old.py   /home/labiiwa/dit-policy/bc_finetune/open/wandb_None_franka_2cam_contact_resnet_gn_2026-06-09_14-25-12/open.ckpt   --gamma 1.0 --T 500 --action_idx 3 --lift_scale 2.0 





