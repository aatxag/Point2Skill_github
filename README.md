# POINT2SKILL

Project web-page: https://point2skill.github.io/Point2Skill-web/




## POLITIKAK EXEKUTATU AHAL IZATEKO

1. Terminala

conda activate data4robotics_py310
source ~/franka_ros2_ws/install/setup.bash
ros2 launch franka_fr3_arm_controllers franka.launch.py arm_id:=fr3 robot_ip:=172.17.6.130 

2. Terminala

source ~/franka_ros2_ws/install/setup.bash
ros2 control load_controller --set-state active joint_impedance_controller

3. Terminala

source ~/franka_ros2_ws/install/setup.bash
ros2 launch franka_gripper gripper.launch.py robot_ip:=172.17.6.130 namespace:=franka_gripper

4. Terminala

ros2 run franka_demo_recorder usb_cam_publisher --ros-args   -p device:=/dev/video6   -p frame_width:=1280   -p frame_height:=960   -p fps:=60   -p topic:=/camera/camera_ext/color/image_raw

5. Terminala

ros2 launch realsense2_camera rs_launch.py     camera_name:=camera_wrist     camera_namespace:=camera     serial_no:="'141122079579'"     enable_color:=true     enable_depth:=true     align_depth.enable:=true 


## Gripperra ireki/itxi nahi bada

source ~/franka_ros2_ws/install/setup.bash
ros2 action send_goal /franka_gripper/franka_gripper/move franka_msgs/action/Move "{width: 0.08, speed: 0.1}"

source ~/franka_ros2_ws/install/setup.bash
ros2 action send_goal /franka_gripper/franka_gripper/grasp franka_msgs/action/Grasp "{width: 0.0, speed: 0.1, force: 20.0, epsilon: {inner: 0.005, outer: 0.03}}"

# Modelo bat bakarrik

python3   eval_franka_2cam_contact.py   /home/labiiwa/Point2Skill_github/dit-policy/bc_finetune/generalization_twoposes/wandb_None_franka_2cam_contact_resnet_gn_2026-06-19_08-48-19/generalization_twoposes.ckpt   --gamma 1.0   --T 500   --action_idx 3   --auto_lift   --grasp_confirm_steps 3   --lift_trigger 0.04

### Open bakarrik adb

python3 eval_franka_2cam_contact.py   /home/labiiwa/Point2Skill_github/dit-policy/bc_finetune/open/wandb_None_franka_2cam_contact_resnet_gn_2026-06-09_14-25-12/open.ckpt   --gamma 1.0   --T 500   --action_idx 3   --lift_scale 2.0   --q_start -0.0714000016450882 -1.264799952507019 0.10849999636411667 -2.9293999671936035 0.1451999992132187 2.075200080871582 -2.3701999187469482

### Hasierako puntua gorde nahi bada

cd dit-policy
cd eval_scripts

python3 eval_franka_2cam_contact_old.py   /home/labiiwa/dit-policy/bc_finetune/close/wandb_None_franka_2cam_contact_resnet_gn_2026-06-12_18-46-27/close.ckpt   --gamma 1.0 --T 500 --action_idx 3 --lift_scale 2.0


# Web-a 

https://point2skill.github.io/Point2Skill-web/

git pull origin main
