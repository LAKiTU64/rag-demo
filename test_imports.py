#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入测试脚本 - 验证所有模块能否正常导入
"""

import sys
from pathlib import Path

print("="*60)
print("AI Agent 模块导入测试")
print("="*60)

# 测试1：检查Python版本
print("\n[测试1] Python版本检查")
print(f"Python版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if sys.version_info >= (3, 8):
    print("[OK] Python version meets requirements (>=3.8)")
else:
    print("[FAIL] Python version too low, need 3.8+")
    sys.exit(1)

# 测试2：检查必要的外部依赖
print("\n[测试2] 外部依赖检查")
dependencies = {
    'fastapi': 'FastAPI Web框架',
    'uvicorn': 'ASGI服务器',
    'yaml': 'YAML配置文件解析',
    'pathlib': 'Path类（标准库）',
}

all_deps_ok = True
for module_name, description in dependencies.items():
    try:
        __import__(module_name)
        print(f"[OK]  {module_name:15} - {description}")
    except ImportError:
        print(f"[FAIL] {module_name:15} - {description} [Not installed]")
        all_deps_ok = False

if not all_deps_ok:
    print("\n[WARNING] Please run: pip install -r requirements.txt")

# 测试3：检查配置文件
print("\n[测试3] 配置文件检查")
config_file = Path("config.yaml")
if config_file.exists():
    print(f"[OK] Config file exists: {config_file}")
    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Config file format is correct")
        print(f"  - sglang_path: {config.get('sglang_path')}")
        print(f"  - models_path: {config.get('models_path')}")
    except Exception as e:
        print(f"[FAIL] Config file parse failed: {e}")
else:
    print(f"[FAIL] Config file not found: {config_file}")

# 测试4：检查目录结构
print("\n[测试4] 目录结构检查")
required_structure = {
    'backend/': '后端代码目录',
    'backend/__init__.py': '后端包初始化',
    'backend/web_server.py': 'Web服务器',
    'backend/agent_core.py': 'AI Agent核心',
    'backend/utils/': '工具模块目录',
    'backend/utils/__init__.py': '工具包初始化',
    'frontend/': '前端目录',
    'frontend/chat.html': '聊天界面',
}

all_structure_ok = True
for path, description in required_structure.items():
    path_obj = Path(path)
    if path_obj.exists():
        print(f"[OK]  {path:30} - {description}")
    else:
        print(f"[FAIL] {path:30} - {description} [Not found]")
        all_structure_ok = False

# 测试5：测试模块导入
print("\n[测试5] 模块导入测试")

# 添加backend到路径
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

# Test import agent_core
try:
    from agent_core import AIAgent
    print("[OK] agent_core.AIAgent imported successfully")
    
    # Test instantiation
    test_config = {
        'sglang_path': 'test_sglang',
        'models_path': 'test_models',
        'model_mappings': {}
    }
    agent = AIAgent(test_config)
    print("[OK] AIAgent instantiated successfully")
    
except ImportError as e:
    print(f"[FAIL] agent_core import failed: {e}")
except Exception as e:
    print(f"[FAIL] AIAgent instantiation failed: {e}")

# 测试6：测试 web_server 的导入（不启动服务）
print("\n[测试6] Web服务器模块检查")
try:
    # Check if web_server.py can be parsed without executing
    import ast
    web_server_file = Path('backend/web_server.py')
    with open(web_server_file, 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("[OK] web_server.py syntax is correct")
except SyntaxError as e:
    print(f"[FAIL] web_server.py syntax error: {e}")
except Exception as e:
    print(f"[FAIL] web_server.py check failed: {e}")

# 测试7：检查 start.py
print("\n[测试7] 启动脚本检查")
try:
    import ast
    start_file = Path('start.py')
    with open(start_file, 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("[OK] start.py syntax is correct")
except SyntaxError as e:
    print(f"[FAIL] start.py syntax error: {e}")
except Exception as e:
    print(f"[FAIL] start.py check failed: {e}")

# Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)

if all_deps_ok and all_structure_ok:
    print("[SUCCESS] All checks passed!")
    print("\nYou can now run: python start.py")
else:
    print("[WARNING] Issues found, please fix according to above")
    if not all_deps_ok:
        print("  1. Run: pip install -r requirements.txt")
    if not all_structure_ok:
        print("  2. Check if file structure is complete")

print("="*60)

