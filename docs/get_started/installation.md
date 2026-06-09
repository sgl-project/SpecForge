# 🚀 Get Started

## 📦 Installation

To install this project, you can simply run the following command.

- **Install from source (recommended)**

```bash
# git clone the source code
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge

# create a new virtual environment
uv venv -p 3.11
source .venv/bin/activate

# install specforge
uv pip install -v . --prerelease=allow
```

- **Install from PyPI**

```bash
pip install specforge
```

- **Install on Ascend NPU**

1. Pull compatible SGLang image for Ascend NPU, currently `quay.io/ascend/sglang:v0.5.9-cann8.5.0-a3` on A3 device, or `quay.io/ascend/sglang:v0.5.9-cann8.5.0-910b` on A2 device.
2. Install SpecForge.

```bash
# git clone the source code
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge

# install specforge
pip install -r requirements-npu.txt
pip install . --no-deps
```
