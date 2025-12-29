# AI Agent LLM性能分析器 - 完整部署包

## 📦 这是什么？

这是一个**完整的、开箱即用的** AI Agent LLM性能分析器部署包。

所有文件已经整理好，路径已经配置正确，可以直接运行。

---

## 🚀 快速开始（3步）

### 第1步：安装依赖

```bash
pip install -r requirements.txt
```

### 第2步：配置必要的路径

打开 `config.yaml` 文件，修改以下内容：

```yaml
# 修改这两个路径为你的实际路径
sglang_path: "D:/Code/sglang"        # ← 你的SGlang代码路径
models_path: "D:/Models"              # ← 你的模型文件路径
```

### 第3步：启动服务

```bash
python start.py
```

然后浏览器打开：**http://localhost:8000/chat**

---

## 📁 目录结构

```
AI_Agent_Complete/
├── start.py                  # 启动脚本（运行这个）
├── config.yaml              # 配置文件（修改这个）
├── requirements.txt         # 依赖列表
├── README.md               # 本文件
├── backend/                # 后端服务
│   ├── web_server.py       # Web服务器
│   ├── agent_core.py       # AI Agent核心
│   └── utils/
│       ├── nsys_parser.py  # NSys解析器
│       └── ncu_parser.py   # NCU解析器
└── frontend/               # 前端界面
    └── chat.html           # 聊天界面
```

---

## ⚙️ 配置说明

### 必须配置的项：

1. **SGlang路径** - `config.yaml` 中的 `sglang_path`
   ```yaml
   sglang_path: "D:/Code/sglang"  # 改为你的路径
   ```

2. **模型路径** - `config.yaml` 中的 `models_path`
   ```yaml
   models_path: "D:/Models"  # 改为你的路径
   ```

### 可选配置：

- **服务器端口**：默认8000，可在 `config.yaml` 修改
- **模型映射**：在 `config.yaml` 中的 `model_mappings` 部分配置

---

## ✅ 运行前检查

运行以下命令检查环境：

```bash
# 检查Python
python --version          # 需要 Python 3.8+

# 检查NVIDIA工具
nvidia-smi               # 检查GPU
nsys --version           # 检查NSight Systems
ncu --version            # 检查NSight Compute

# 检查依赖
pip list | findstr "fastapi pandas"
```

---

## 🎯 使用示例

### 启动服务后：

1. 浏览器打开 http://localhost:8000/chat
2. 在对话框输入：
   ```
   分析 llama-7b 模型，batch_size=8
   ```
3. AI会自动解析并开始分析

### 支持的命令格式：

```
分析 llama-7b，batch_size=8,16
对 qwen-14b 进行 nsys 全局分析
综合分析 chatglm-6b 的性能瓶颈
使用 ncu 深度分析 vicuna-7b
```

### 🆕 前端 API 选择器

- 聊天输入框上方新增了 **API 下拉框**，可快速切换不同的分析后端：
   - `智能推荐（自动选择）`：默认策略，优先调用 LangChain Agent，如未配置则回退为 NSys 示例流程；
   - `LangChain Agent`：直接触发 `agent_core.AIAgent`，用于完整的智能分析流程；
   - `NSys 性能分析 / NCU 深度分析`：返回针对 Nsight Systems 与 Nsight Compute 的操作指引；
   - `自定义工具链`：保留扩展位，可在 `backend/web_server.py` 的 `dispatch_api_request` 中新增逻辑；
- 选择会随着消息一同发送到后端，可在控制台/日志中确认路由是否正确。

### 🧠 向量知识库示例脚本

`backend/knowledge_bases/` 目录新增两份脚本，便于日后将性能报告或模型知识写入向量库：

| 脚本 | 功能 | 运行方式 |
| --- | --- | --- |
| `faiss_in_memory_kb.py` | 构建内存型 FAISS 向量库，演示检索与导出 | `python backend/knowledge_bases/faiss_in_memory_kb.py` |
| `persistent_chroma_kb.py` | 使用 Chroma 创建可持久化的向量库并重新加载 | `python backend/knowledge_bases/persistent_chroma_kb.py` |

> 📌 若首次使用 LangChain 相关功能，请参考脚本顶部的依赖安装说明（`langchain-community`、`sentence-transformers`、`chromadb` 等）。

### 🔗 NSys → NCU 内核名称提取与深度分析流程

新增脚本：`backend/utils/extract_nsys_kernels.py` 用于从 `.nsys-rep` 中自动抽取热点 CUDA kernel 名（按总耗时排序），生成可直接用于 NCU 的 `--kernel-name` 参数列表。

使用步骤：
```bash
# 1. 运行 nsys 全局分析（示例）
nsys profile -o run_profile -t cuda,nvtx,osrt --force-overwrite=true python your_program.py

# 2. 提取热点 kernel 名称
python backend/utils/extract_nsys_kernels.py --rep run_profile.nsys-rep --top-k 8 --min-avg-ms 0.05 --out kernels.txt

# 3. 查看结果
cat kernels.txt

# 4. 针对前几个 kernel 做 NCU 深度分析（可先精确匹配，失败再用 regex 前缀）
ncu --kernel-name "$(sed -n '1p' kernels.txt)" \
      --kernel-name "$(sed -n '2p' kernels.txt)" \
      --set full -o ncu_hotkernels -- python your_program.py
```

说明：
- 如果提取出的名称为数字或 `__unnamed_` 开头，可加 `--include-placeholder` 保留，再通过 `ncu --list-kernels` 发现真实名后替换。
- Hopper (SM 9.0) 上大型 CUTLASS / FlashAttention kernel 名较长，精确匹配失败时可改用：
   ```bash
   ncu --kernel-name "regex:^void Kernel2<cutlass_80_simt_sgemm" --set full -o ncu_hotkernels -- python your_program.py
   ```
- 低 Occupancy 的 GEMM 不一定是问题（受寄存器+SMEM 限制的 compute-bound），请结合 `Compute Throughput` 与 `Issue Slots Busy` 进行判断。

常见排错：
- `ncu` 无输出：确认 kernel 名是否精确匹配（尝试 demangled 前缀）
- 生成多 pass 报告：属正常；某些指标需拆分采集
- Dropped Samples 较多：减少采集集合或降低采样频率

---

## 🐛 常见问题

### 问题1：启动失败

**检查**：
```bash
# 查看详细错误
python start.py
```

**可能原因**：
- 依赖未安装：运行 `pip install -r requirements.txt`
- 端口被占用：修改 `config.yaml` 中的 `port`

### 问题2：找不到模型

**检查**：
- `config.yaml` 中的 `models_path` 是否正确
- 模型文件是否存在
- `model_mappings` 配置是否正确

### 问题3：SGlang命令执行失败

**检查**：
- `config.yaml` 中的 `sglang_path` 是否正确
- SGlang是否已安装：`cd <sglang_path> && python -m sglang.launch_server --help`

---

## 📞 获取帮助

1. 查看日志：运行 `start.py` 时的输出
2. 检查配置：`cat config.yaml`
3. 测试连接：`curl http://localhost:8000/health`

---

## 🔄 更新日志

- v1.0.0: 初始版本，整理完整部署包
- 包含路径修复
- 统一配置文件
- 简化启动流程

---

## 💡 下一步

成功运行后，你可以：

1. 上传配置文件进行分析
2. 查看生成的性能报告
3. 根据建议优化模型性能

祝使用愉快！🎉

