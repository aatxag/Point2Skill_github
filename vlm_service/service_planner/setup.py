from glob import glob

from setuptools import find_packages, setup

package_name = "service_planner"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="labiiwa",
    maintainer_email="aatxag@mondragon.edu",
    description="VLM-based task planner mapping language to diffusion policies",
    license="MIT",
    entry_points={
        "console_scripts": [
            "planner = service_planner.planner:main",
        ],
    },
)
