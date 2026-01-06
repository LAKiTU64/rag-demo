# 使用外部镜像源
# FROM docker.io/lmsysorg/sglang:v0.4.7-cu124

# 使用指定的华为云镜像作为基础（内网可用）
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:v0.4.7-cu124

# 切换到 root 用户进行软件包安装
USER root

# 替换 Debian 的默认 APT 源为阿里云镜像
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 更新软件包列表并安装必要的工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置 pip 使用清华源
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Python 依赖
COPY requirements.txt /tmp/
RUN pip3 install --break-system-packages -r /tmp/requirements.txt

# 清理缓存
RUN pip cache purge