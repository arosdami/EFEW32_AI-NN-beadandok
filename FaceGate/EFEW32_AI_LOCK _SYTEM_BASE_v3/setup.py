# setup.py
from setuptools import setup, find_packages
import sys

setup(
    name="AdvancedFaceLockSystem",
    version="1.0.0",
    description="Professional AI Face Recognition Security System",
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.5.0',
        'tensorflow>=2.8.0',
        'numpy>=1.21.0',
        'pyserial>=3.5',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'facelock=main:main',
        ],
    },
    include_package_data=True,
)