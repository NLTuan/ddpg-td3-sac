# ddpg-td3-sac
Continuous action Q-Learning one big family

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable package management. PyTorch dependencies are configured via optional dependency groups to prevent conflicts between CPU and GPU versions.

### 1. Prerequisites
Ensure you have `uv` installed. If you don't have it, install it using:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Environment Installation

Depending on your target device, run the corresponding command. This will create a virtual environment (`.venv`), install all core requirements (like Gymnasium), and resolve the appropriate PyTorch wheels for your hardware.

**For CPU-only environments:**
```bash
uv sync --extra cpu
```

**For GPU (CUDA 12.4) environments:**
```bash
uv sync --extra gpu
```

*Note: Do not run both commands at the same time or chain the extras (`--extra cpu --extra gpu`), as the sources are explicitly configured to be mutually exclusive to avoid package clashing.*
