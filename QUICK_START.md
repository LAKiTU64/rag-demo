# 快速开始指南

## 🎯 3步启动

### 1. 安装依赖
```bash
cd AI_Agent_Complete
pip install -r requirements.txt
```

### 2. 修改配置
打开 `config.yaml`，修改这两行：
```yaml
sglang_path: "D:/Code/sglang"    # ← 改成你的SGlang路径
models_path: "D:/Models"          # ← 改成你的模型路径
```

### 3. 启动服务
```bash
python start.py
```

浏览器打开：**http://localhost:8000/chat**

---

## ✅ 验证运行

### 检查服务状态
```bash
# 访问健康检查接口
curl http://localhost:8000/health
```

预期输出：
```json
{
  "status": "healthy",
  "agent_ready": true,
  "config_loaded": true
}
```

### 测试对话
在聊天界面输入：
```
分析 llama-7b 模型，batch_size=8
```

应该看到AI的解析回复。

---

## 📁 目录说明

```
AI_Agent_Complete/
├── start.py          # 👈 运行这个启动服务
├── config.yaml       # 👈 修改这个配置路径
├── requirements.txt  # 依赖列表
├── README.md         # 完整文档
├── backend/          # 后端代码
│   ├── web_server.py
│   ├── agent_core.py
│   └── utils/
└── frontend/         # 前端界面
    └── chat.html
```

---

## ⚙️ 配置示例

### 示例1：本地开发
```yaml
sglang_path: "D:/Code/sglang"
models_path: "D:/Models"

model_mappings:
  "llama-7b": "Llama-2-7b-hf"      # 相对于models_path
  "qwen-14b": "Qwen-14B-Chat"
```

### 示例2：使用绝对路径
```yaml
sglang_path: "D:/Code/sglang"
models_path: "D:/Models"

model_mappings:
  "llama-7b": "D:/Models/Llama-2-7b-hf"    # 绝对路径
  "qwen-14b": "E:/LLMs/Qwen-14B-Chat"      # 可以跨盘符
```

### 示例3：使用HuggingFace ID
```yaml
model_mappings:
  "llama-7b": "meta-llama/Llama-2-7b-hf"   # 会自动下载
```

---

## 🐛 常见问题

### 问题1：启动失败
```
❌ 错误: 找不到 backend/web_server.py
```
**解决**：确保在 AI_Agent_Complete 目录下运行 `python start.py`

### 问题2：依赖缺失
```
❌ 错误: No module named 'fastapi'
```
**解决**：运行 `pip install -r requirements.txt`

### 问题3：端口被占用
```
❌ 错误: Address already in use
```
**解决**：修改 `config.yaml` 中的 `port: 8000` 改为其他端口，如 `8080`

### 问题4：找不到模型
```
⚠️ 警告: 模型路径不存在
```
**解决**：
1. 检查 `config.yaml` 中的 `models_path` 是否正确
2. 确认模型文件已下载到该目录
3. 检查 `model_mappings` 中的路径配置

---

## 🚀 下一步

启动成功后，你可以：

1. **测试基本功能**
   - 在对话框输入分析请求
   - 上传配置文件
   - 查看AI的解析结果

2. **配置完整功能**
   - 安装SGlang：`git clone https://github.com/sgl-project/sglang.git`
   - 下载模型文件
   - 安装分析工具（nsys, ncu）

3. **进行性能分析**
   - 运行nsys全局分析
   - 执行ncu深度分析
   - 查看生成的报告

---

## 📞 获取帮助

- 查看完整文档：`README.md`
- 检查配置：`cat config.yaml`
- 查看日志：运行 `python start.py` 时的输出

---

祝使用愉快！🎉

