# AI Agent 部署检查清单

## ✅ 部署前检查

### 1. 环境检查
- [ ] Python 3.8+ 已安装
  ```bash
  python --version
  ```

- [ ] pip 可用
  ```bash
  pip --version
  ```

- [ ] GPU和驱动正常（可选，用于性能分析）
  ```bash
  nvidia-smi
  ```

### 2. 依赖安装
- [ ] 安装Python依赖
  ```bash
  cd AI_Agent_Complete
  pip install -r requirements.txt
  ```

- [ ] 验证核心依赖
  ```bash
  python -c "import fastapi, uvicorn, yaml; print('✓ 核心依赖OK')"
  ```

### 3. 配置修改
- [ ] 打开 `config.yaml`
- [ ] 修改 `sglang_path` 为你的SGlang路径
- [ ] 修改 `models_path` 为你的模型路径
- [ ] （可选）修改 `model_mappings`
- [ ] （可选）修改服务器端口

### 4. 文件完整性
运行以下命令检查文件：
```bash
python -c "
from pathlib import Path
files = [
    'start.py',
    'config.yaml',
    'requirements.txt',
    'backend/web_server.py',
    'backend/agent_core.py',
    'frontend/chat.html'
]
for f in files:
    status = '✓' if Path(f).exists() else '✗'
    print(f'{status} {f}')
"
```

所有文件应显示 ✓

---

## 🚀 启动流程

### 步骤1：进入目录
```bash
cd AI_Agent_Complete
```

### 步骤2：检查配置
```bash
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
```

### 步骤3：启动服务
```bash
python start.py
```

### 步骤4：验证服务
在新的终端窗口运行：
```bash
curl http://localhost:8000/health
```

预期输出：
```json
{
  "status": "healthy",
  "timestamp": "...",
  "active_connections": 0,
  "agent_ready": true,
  "config_loaded": true
}
```

### 步骤5：访问前端
浏览器打开：http://localhost:8000/chat

---

## 🧪 功能测试

### 测试1：基本对话
在聊天界面输入：
```
你好
```

应该看到欢迎消息。

### 测试2：模型解析
在聊天界面输入：
```
分析 llama-7b 模型，batch_size=8
```

应该看到AI解析的参数信息。

### 测试3：文件上传
1. 点击左侧的文件上传区域
2. 上传一个JSON或YAML配置文件
3. 查看AI的解析结果

### 测试4：API接口
```bash
# 检查配置接口
curl http://localhost:8000/config

# 检查健康状态
curl http://localhost:8000/health
```

---

## 📊 目录结构验证

运行以下命令查看目录结构：
```bash
tree /F AI_Agent_Complete
```

或在PowerShell中：
```powershell
tree /F AI_Agent_Complete
```

应该看到：
```
AI_Agent_Complete/
├── start.py
├── config.yaml
├── requirements.txt
├── README.md
├── QUICK_START.md
├── DEPLOYMENT_CHECKLIST.md
├── backend/
│   ├── __init__.py
│   ├── web_server.py
│   ├── agent_core.py
│   └── utils/
│       └── __init__.py
└── frontend/
    └── chat.html
```

---

## 🔧 高级配置（可选）

### 配置SGlang服务
如果有本地SGlang服务器：
```yaml
# config.yaml
sglang_service:
  host: "192.168.1.100"  # SGlang服务器IP
  port: 30000             # SGlang服务端口
```

### 配置分析工具
如果安装了nsys和ncu：
```yaml
# config.yaml
profiling_tools:
  nsys:
    enabled: true
    timeout: 600
  ncu:
    enabled: true
    timeout: 600
    max_kernels: 5
```

### 修改服务器配置
```yaml
# config.yaml
server:
  host: "0.0.0.0"    # 允许外部访问
  port: 8080          # 修改端口
```

---

## 🐛 故障排除

### 问题：模块导入错误
```
ImportError: No module named 'xxx'
```
**解决**：
```bash
pip install -r requirements.txt --upgrade
```

### 问题：配置文件错误
```
yaml.scanner.ScannerError: ...
```
**解决**：
1. 检查 `config.yaml` 格式
2. 确保缩进使用空格而非Tab
3. 确保字符串用引号包围

### 问题：端口被占用
```
OSError: [Errno 98] Address already in use
```
**解决**：
```bash
# 查看占用端口的进程
netstat -ano | findstr :8000
# 或修改config.yaml中的端口
```

### 问题：无法访问前端
```
404 Not Found
```
**解决**：
1. 确认 `frontend/chat.html` 存在
2. 重启服务
3. 清除浏览器缓存

---

## 📝 部署记录

完成部署后，记录以下信息：

- [ ] 部署日期：__________
- [ ] Python版本：__________
- [ ] 服务器地址：__________
- [ ] 服务端口：__________
- [ ] SGlang路径：__________
- [ ] 模型路径：__________
- [ ] 测试状态：□ 通过 □ 失败

---

## 🎉 部署完成

如果所有检查都通过，恭喜你！AI Agent已经成功部署。

现在可以：
1. 开始使用聊天界面
2. 测试性能分析功能
3. 查看生成的报告

祝使用愉快！

