# Copyright (c) 2024，Zhangzijie.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_path = get_package_share_directory('mipi_detect')
    yolo_detect_path = os.path.join(package_path, 'lib/mipi_detect')
    config_src = os.path.join(yolo_detect_path, 'config')
    config_dst = os.path.join(os.getcwd(), 'config')
    os.system(f"cp -r {config_src} {config_dst}")

    # 关闭HDMI默认界面显示
    os.system("sudo systemctl stop lightdm")
    
    param_file = os.path.join(config_dst, 'params.yaml')

    return LaunchDescription([
        Node(
            package='mipi_detect',
            executable='mipi_detect_node',
            output='screen',
            parameters=[param_file],
            arguments=['--ros-args', '--log-level', 'info']
        )
    ])