from setuptools import setup, find_packages


VERSION = "0.0.1"

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")


setup(
    name="eisen-extras",
    version=VERSION,
    description=(
        "Extra functionality and foreign functionality adapted for use within Eisen"
    ),
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.6",
    entry_points={"console_scripts": [],},
)
