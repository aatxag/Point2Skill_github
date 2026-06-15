from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    python_path_arg = DeclareLaunchArgument(
        "python_path",
        default_value="/home/labiiwa/venvs/ml/bin/python3",
    )
    topic_selected_policy = DeclareLaunchArgument(
        "topic_selected_policy",
        default_value="/selected_policy",
    )
    topic_policy_stop = DeclareLaunchArgument(
        "topic_policy_stop",
        default_value="/policy_stop",
    )
    topic_policy_status = DeclareLaunchArgument(
        "topic_policy_status",
        default_value="/policy_execution_status",
    )

    cmd = [
        LaunchConfiguration("python_path"),
        "-m", "service_primitives.primitives",
        "--ros-args",
        "-p", ["topic_selected_policy:=", LaunchConfiguration("topic_selected_policy")],
        "-p", ["topic_policy_stop:=",     LaunchConfiguration("topic_policy_stop")],
        "-p", ["topic_policy_status:=",   LaunchConfiguration("topic_policy_status")],
    ]

    return LaunchDescription([
        python_path_arg,
        topic_selected_policy,
        topic_policy_stop,
        topic_policy_status,
        ExecuteProcess(cmd=cmd, output="screen"),
    ])
