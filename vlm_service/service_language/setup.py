from setuptools import setup, find_packages

package_name = "service_language"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/language.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="labiiwa",
    maintainer_email="aatxag@mondragon.edu",
    description="ROS2 port of service_language",
    license="MIT",
    entry_points={
        "console_scripts": [
            "language_publisher = service_language.language_publisher:main",
            "language_publisher_answer = service_language.language_publisher_answer:main",
        ],
    },
)
