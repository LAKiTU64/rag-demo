# # 推荐使用 NVIDIA 官方提供的 Python 环境，省去配置 CUDA 的烦恼
# FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# # 安装 Python 和基础工具
# RUN apt-get update && apt-get install -y \
#     python3.11 \
#     python3-pip \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /workspaces/${localWorkspaceFolderBasename}


FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.0-base-ubuntu22.04


# 配置阿里云镜像源并安装系统依赖
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    git \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/${localWorkspaceFolderBasename}