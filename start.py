#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent LLM性能分析器 - 启动脚本

快速启动AI Agent服务
"""

import sys
import subprocess
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要3.8+")
        print(f"   当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # 检查必要文件
    required_files = [
        "config.yaml",
        "requirements.txt",
        "backend/web_server.py",
        "frontend/chat.html"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("✅ 必要文件检查通过")
    
    # 检查依赖
    try:
        import fastapi
        import uvicorn
        import yaml
        print("✅ 核心依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("   请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_config():
    """检查配置文件"""
    print("\n⚙️ 检查配置...")
    
    try:
        import yaml
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置
        if 'sglang_path' not in config:
            print("⚠️ 配置文件缺少 sglang_path")
        else:
            print(f"   SGlang路径: {config['sglang_path']}")
        
        if 'models_path' not in config:
            print("⚠️ 配置文件缺少 models_path")
        else:
            print(f"   模型路径: {config['models_path']}")
        
        print("✅ 配置文件格式正确")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件错误: {e}")
        return False

def start_server():
    """启动服务器"""
    print("\n🚀 启动AI Agent服务...")
    print("="*60)
    
    # 读取配置获取端口
    try:
        import yaml
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        port = config.get('server', {}).get('port', 8000)
        host = config.get('server', {}).get('host', '0.0.0.0')
        
        print(f"\n📡 服务地址: http://localhost:{port}")
        print(f"💬 聊天界面: http://localhost:{port}/chat")
        print(f"📚 API文档: http://localhost:{port}/docs")
        print("\n按 Ctrl+C 停止服务")
        print("="*60)
        print()
        
        # 启动服务
        backend_path = Path("backend/web_server.py")
        subprocess.run([sys.executable, str(backend_path)], check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except FileNotFoundError:
        print("❌ 找不到 backend/web_server.py")
        print("   请确保文件存在")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════╗
║   AI Agent LLM性能分析器                                 ║
║   版本: 1.0.0                                            ║
╚══════════════════════════════════════════════════════════╝
""")
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请先解决上述问题")
        sys.exit(1)
    
    # 检查配置
    if not check_config():
        print("\n❌ 配置检查失败，请检查 config.yaml")
        sys.exit(1)
    
    # 启动服务
    start_server()

if __name__ == "__main__":
    main()

