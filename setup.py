from setuptools import find_packages, setup


def read_requirements():
    with open(f"requirements.txt", "r") as f:
        lines = (line.strip() for line in f)
        return [line for line in lines if line and not line.startswith(("#", "--"))]


def read_readme():
    with open("README.md", "r") as f:
        return f.read()


def read_version():
    with open("version.txt", "r") as f:
        return f.read().strip()


setup(
    name="specforge",
    packages=find_packages(exclude=["configs", "scripts", "tests"]),
    version=read_version(),
    install_requires=read_requirements(),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="SGLang Team",
    url="https://github.com/sgl-project/SpecForge",
)
