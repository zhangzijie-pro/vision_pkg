from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision_pkg_deepsort',
            executable='track_and_control',
            name='track_and_control',
            output='screen',
            parameters=[
                {'target_class': 0},
                {'conf_thres': 0.35},
                {'cam_id': 0},
                {'width': 960},
                {'height': 544},
                {'visualize': False},
                {'embedder': 'mobilenet'},
            ]
        )
    ])
