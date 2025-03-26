from setuptools import find_packages
from distutils.core import setup

setup(
    name='sim2sim_leggedgym',
    version='1.0.0',
    author='IUST LAB',
    license="IUST LAB",
    packages=find_packages(),
    author_email='mmhdimansouri@gmail.com',
    description='Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'matplotlib',
                      'tqdm',
                      'numpy==1.23.5',
                      'opencv-python',
                      'mujoco==2.3.6',
                      'mujoco-python-viewer']
)
