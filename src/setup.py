from setuptools import setup, find_packages
setup(
    name="ava_agent",
    version="0.1.0",
    package_dir={"":"ava_agent"},
    packages=find_packages(where="ava_agent")
)