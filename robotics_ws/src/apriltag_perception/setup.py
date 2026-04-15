from setuptools import find_packages, setup

package_name = 'apriltag_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoheen',
    maintainer_email='yoheen@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'apriltag_publisher = apriltag_perception.apriltag_publisher:main',
            'apriltag_subscriber = apriltag_perception.apriltag_subscriber:main',
            'pose_publisher = apriltag_perception.pose_publisher:main',
            'gesture_collector = apriltag_perception.gesture_collector:main',
        ],
    },
)
