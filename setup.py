from setuptools import setup

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name='ml-engine',
    version='1.0.0',
    author='Manh Tu VU',
    description='ML engine functions',
    packages=['ml_engine'],
    install_requires=requirements,
)