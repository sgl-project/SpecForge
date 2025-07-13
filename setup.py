from setuptools import find_packages, setup


def read_requirements():
    with open(f"requirements.txt", "r") as f:
        return f.read().splitlines()


def read_readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="sgl-spec",
    packages=find_packages(exclude=["configs", "scripts", "tests"]),
    version="0.0.1",
    install_requires=read_requirements(),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="sgl-project",
    url="https://github.com/sgl-project",
)
