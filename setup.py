from setuptools import find_packages, setup


def read_requirements():
    with open(f"requirements.txt", "r") as f:
        return f.read().splitlines()


def read_readme():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="specforge",
    packages=find_packages(exclude=["configs", "scripts", "tests"]),
    version="0.1.0",
    install_requires=read_requirements(),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="SGLang Team",
    url="https://github.com/sgl-project/SpecForge",
)
