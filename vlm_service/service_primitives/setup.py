from setuptools import setup, find_packages
from glob import glob

package_name = 'service_primitives'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='labiiwa',
    maintainer_email='aatxag@mondragon.edu',
    description='Diffusion policy launcher primitive',
    license='MIT',
    entry_points={
        'console_scripts': [
            'primitives = service_primitives.primitives:main',
        ],
    },
)
