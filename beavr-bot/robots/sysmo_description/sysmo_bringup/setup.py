from setuptools import setup

package_name = 'sysmo_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=['sysmo_bringup'],
    install_requires=['setuptools', 'rclpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A description of your package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'test_sysmo_hw = sysmo_bringup.sysmo_bringup.test_sysmo_hw:main',  # 这是你Python脚本的入口点
        ],
    },
)

