from glob import glob
from setuptools import setup

package_name = "sysmo_rviz_command_bridge"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="likunwei",
    maintainer_email="likunwei@example.com",
    description="Bridge SYSMO-32 18D arm commands into interpolated JointState messages for RViz.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sysmo32_command_to_joint_state = "
            "sysmo_rviz_command_bridge.command_joint_state_bridge:main",
        ],
    },
)
