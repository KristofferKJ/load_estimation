from setuptools import find_packages, setup

package_name = 'load_estimation'

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
    maintainer='rasmus-storm',
    maintainer_email='rasmus@stormtrooper.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'load_estimation_node = load_estimation.load_estimation:main',
            'gimbal_control_node = load_estimation.gimbal_control:main',
            #'camera_driver_node = load_estimation.camera_driver:main',
        ],
    },
)
