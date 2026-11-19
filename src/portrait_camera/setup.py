import glob
import os

from setuptools import find_packages, setup

package_name = 'portrait_camera'


def data(pattern, dest):
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    return [(os.path.join('share', package_name, dest), files)] if files else []


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + data('config/*', 'config') + data('models/*', 'models'),
    install_requires=['setuptools'],
    zip_safe=True,
    description='Webcam node (OpenCV) publishing images and camera info; swappable for a DSLR.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = portrait_camera.camera_node:main',
        ],
    },
)
