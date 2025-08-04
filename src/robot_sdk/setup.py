from setuptools import find_packages, setup

package_name = 'robot_sdk'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'scipy'],
    zip_safe=True,
    maintainer='andrin',
    maintainer_email='AndrinWinzap@proton.me',
    description='Python library to control a 6dof robot arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
