from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("python_path",           default_value="/home/labiiwa/venvs/ml/bin/python3"),
        DeclareLaunchArgument("topic_prompt",          default_value="/topic_prompt"),
        DeclareLaunchArgument("topic_response",        default_value="/planner_response"),
        DeclareLaunchArgument("topic_rgb_wrist",       default_value="/camera/camera_wrist/color/image_raw"),
        DeclareLaunchArgument("topic_selected_policy", default_value="/selected_policy"),
        DeclareLaunchArgument("topic_policy_status",   default_value="/policy_execution_status"),
        DeclareLaunchArgument("topic_object_centroid", default_value="/object_centroid"),
        DeclareLaunchArgument("policy_timeout",        default_value="120.0"),
        DeclareLaunchArgument("headless",              default_value="False"),
        DeclareLaunchArgument("confirm_plan",          default_value="True"),
        DeclareLaunchArgument("enable_localization",   default_value="False"),
        DeclareLaunchArgument("tmp_image_dir",         default_value="/tmp/planner_locate"),

        ExecuteProcess(
            cmd=[
                LaunchConfiguration("python_path"),
                "-m", "service_planner.planner",
                "--ros-args",
                "-p", ["topic_prompt:=",          LaunchConfiguration("topic_prompt")],
                "-p", ["topic_response:=",        LaunchConfiguration("topic_response")],
                "-p", ["topic_rgb_wrist:=",       LaunchConfiguration("topic_rgb_wrist")],
                "-p", ["topic_selected_policy:=", LaunchConfiguration("topic_selected_policy")],
                "-p", ["topic_policy_status:=",   LaunchConfiguration("topic_policy_status")],
                "-p", ["topic_object_centroid:=", LaunchConfiguration("topic_object_centroid")],
                "-p", ["policy_timeout:=",        LaunchConfiguration("policy_timeout")],
                "-p", ["headless:=",              LaunchConfiguration("headless")],
                "-p", ["confirm_plan:=",          LaunchConfiguration("confirm_plan")],
                "-p", ["enable_localization:=",   LaunchConfiguration("enable_localization")],
                "-p", ["tmp_image_dir:=",         LaunchConfiguration("tmp_image_dir")],
            ],
            output="screen",
        ),
    ])
