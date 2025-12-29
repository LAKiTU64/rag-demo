#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试分析工具集成
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试导入"""
    print("🧪 测试模块导入...")
    
    try:
        from backend.utils.nsys_parser import NsysParser, NsysAnalyzer
        print("✅ nsys_parser 导入成功")
    except Exception as e:
        print(f"❌ nsys_parser 导入失败: {e}")
        return False
    
    try:
        from backend.utils.ncu_parser import NCUParser, NCUAnalyzer
        print("✅ ncu_parser 导入成功")
    except Exception as e:
        print(f"❌ ncu_parser 导入失败: {e}")
        return False
    
    try:
        from backend.utils.nsys_to_ncu_analyzer import NSysToNCUAnalyzer
        print("✅ nsys_to_ncu_analyzer 导入成功")
    except Exception as e:
        print(f"❌ nsys_to_ncu_analyzer 导入失败: {e}")
        return False
    
    try:
        from backend.agent_core import AIAgent
        print("✅ agent_core 导入成功")
    except Exception as e:
        print(f"❌ agent_core 导入失败: {e}")
        return False
    
    return True

def test_agent_initialization():
    """测试Agent初始化"""
    print("\n🧪 测试Agent初始化...")
    
    try:
        import yaml
        from backend.agent_core import AIAgent
        
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        agent = AIAgent(config)
        print(f"✅ Agent初始化成功")
        print(f"   - 模型路径: {agent.models_path}")
        print(f"   - SGlang路径: {agent.sglang_path}")
        print(f"   - 结果目录: {agent.results_dir}")
        print(f"   - 可用模型: {', '.join(agent.get_available_models())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_message_parsing():
    """测试消息解析"""
    print("\n🧪 测试消息解析...")
    
    try:
        import yaml
        from backend.agent_core import AIAgent
        
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        agent = AIAgent(config)
        
        # 测试用例
        test_messages = [
            "分析 llama-7b 模型，batch_size=8",
            "对 qwen-14b 进行 nsys 全局分析",
            "使用 ncu 深度分析 chatglm-6b，batch_size=16，input_len=512",
        ]
        
        for msg in test_messages:
            print(f"\n📝 测试消息: '{msg}'")
            model_name = agent._extract_model_name(msg)
            analysis_type = agent._extract_analysis_type(msg)
            params = agent._extract_parameters(msg)
            
            print(f"   - 模型: {model_name}")
            print(f"   - 分析类型: {analysis_type}")
            print(f"   - 参数: {params}")
        
        print("✅ 消息解析测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 消息解析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """测试依赖库"""
    print("\n🧪 测试依赖库...")
    
    dependencies = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('yaml', 'PyYAML'),
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
    ]
    
    all_ok = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name} 已安装")
        except ImportError:
            print(f"❌ {package_name} 未安装 - 运行: pip install {package_name}")
            all_ok = False
    
    return all_ok

def check_nvidia_tools():
    """检查NVIDIA工具"""
    print("\n🧪 检查NVIDIA工具...")
    
    import subprocess
    
    tools = [
        ('nsys', 'NVIDIA Nsight Systems'),
        ('ncu', 'NVIDIA Nsight Compute'),
        ('nvidia-smi', 'NVIDIA GPU Driver'),
    ]
    
    for cmd, name in tools:
        try:
            result = subprocess.run(
                [cmd, '--version'], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ {name}: {version_line}")
            else:
                print(f"⚠️  {name}: 可能未正确安装")
        except FileNotFoundError:
            print(f"❌ {name}: 未找到 - 请安装并添加到PATH")
        except subprocess.TimeoutExpired:
            print(f"⚠️  {name}: 检查超时")
        except Exception as e:
            print(f"⚠️  {name}: 检查失败 - {e}")

def main():
    """主测试函数"""
    print("="*60)
    print("🔧 AI Agent 完整性测试")
    print("="*60)
    
    # 测试1: 导入
    test1 = test_imports()
    
    # 测试2: 依赖
    test2 = test_dependencies()
    
    # 测试3: Agent初始化
    test3 = test_agent_initialization()
    
    # 测试4: 消息解析
    test4 = test_message_parsing()
    
    # 检查NVIDIA工具
    check_nvidia_tools()
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    print(f"导入测试: {'✅ 通过' if test1 else '❌ 失败'}")
    print(f"依赖测试: {'✅ 通过' if test2 else '⚠️  部分缺失'}")
    print(f"初始化测试: {'✅ 通过' if test3 else '❌ 失败'}")
    print(f"消息解析测试: {'✅ 通过' if test4 else '❌ 失败'}")
    
    if test1 and test3 and test4:
        print("\n🎉 所有核心功能测试通过！")
        print("\n💡 下一步:")
        print("1. 配置 config.yaml 中的 sglang_path 和 models_path")
        print("2. 确保安装了 nsys 和 ncu 工具")
        print("3. 运行 python start.py 启动服务")
        print("4. 访问 http://localhost:8000/chat 开始分析")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())





