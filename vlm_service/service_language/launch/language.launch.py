from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="service_language",
            executable="language_publisher",
            name="language_publisher",
            output="screen",
        )
    ])
