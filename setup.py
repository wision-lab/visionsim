from setuptools import find_packages, setup

# TODO: Add proper metadata
setup(
    name="spsim",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
