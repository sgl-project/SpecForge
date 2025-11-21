import tomllib
from pathlib import Path

from setuptools import find_packages, setup


def read_readme():
    with open("README.md", "r") as f:
        return f.read()


def read_version():
    with open("version.txt", "r") as f:
        return f.read().strip()


def read_dependencies():
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
        return pyproject.get("project", {}).get("dependencies", [])


setup(
    name="specforge",
    packages=find_packages(exclude=["configs", "scripts", "tests"]),
    version=read_version(),
    install_requires=read_dependencies(),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="SGLang Team",
    url="https://github.com/sgl-project/SpecForge",
)
