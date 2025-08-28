from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'mipi_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),

        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mipi_detect_node = mipi_detect.detect_node:main',
            'track_deepsort = mipi_detect.track_deepsort:main',
            'mutil_thread_crop_detect_node = mipi_detect.detect_crop_thread_node:main'
        ],
    },
)
