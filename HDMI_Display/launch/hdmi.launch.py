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
    package_path = get_package_share_directory('HDMI_Display')
    HDMI_Display_path = os.path.join(package_path, 'lib/HDMI_Display')
    config_src = os.path.join(HDMI_Display_path, 'config')
    config_dst = os.path.join(os.getcwd(), 'config')
    os.system(f"cp -r {config_src} {config_dst}")
    
    # 指定 YAML 配置文件路径
    param_file = os.path.join(config_dst, 'params.yaml')

    return LaunchDescription([
        Node(
            package='HDMI_Display',
            executable='hdmi_node',
            output='screen',
            parameters=[param_file],
            arguments=['--ros-args', '--log-level', 'info']
        )
    ])