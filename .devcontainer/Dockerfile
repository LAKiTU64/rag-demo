# 推荐使用 NVIDIA 官方提供的 Python 环境，省去配置 CUDA 的烦恼
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 安装 Python 和基础工具
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/${localWorkspaceFolderBasename}