from setuptools import setup, find_packages

package_name = 'vision_pkg_deepsort'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/track_and_control.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhangzijie-pro',
    maintainer_email='',
    description='RDK X5 DeepSORT lock-on tracking with libsrcampy camera',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'track_and_control = vision_pkg_deepsort.track_and_control_node:main',
        ],
    },
)
