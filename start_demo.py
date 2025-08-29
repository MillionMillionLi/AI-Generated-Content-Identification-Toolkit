#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态水印工具 Web Demo 启动脚本
简化的启动流程，包含环境检查和依赖安装指导
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🔍 检查Python版本...")
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        sys.exit(1)
    print(f"✅ Python版本: {sys.version}")

def check_dependencies():
    """检查基础依赖"""
    print("\n🔍 检查基础依赖...")
    
    missing_packages = []
    basic_packages = ['flask', 'flask_cors', 'torch', 'transformers']
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("\n请先安装依赖:")
        print("1. 安装Web依赖: pip install -r requirements_web.txt")
        print("2. 安装核心依赖: pip install -r requirements.txt")
        return False
    
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n🔍 检查项目结构...")
    
    required_files = [
        'src/unified/watermark_tool.py',
        'templates/index.html',
        'app.py'
    ]
    
    project_root = Path(__file__).parent
    missing_files = []
    
    for file in required_files:
        file_path = project_root / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ 缺少以下文件: {', '.join(missing_files)}")
        return False
    
    return True

def test_watermark_tool():
    """测试WatermarkTool是否可以正常导入"""
    print("\n🔍 测试水印工具导入...")
    
    try:
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "src"))
        
        from src.unified.watermark_tool import WatermarkTool
        print("✅ WatermarkTool 导入成功")
        
        # 尝试初始化（但不强制成功）
        try:
            tool = WatermarkTool()
            print("✅ WatermarkTool 初始化成功")
            return True
        except Exception as e:
            print(f"⚠️ WatermarkTool 初始化失败: {e}")
            print("   这可能是由于模型文件缺失，但Web服务仍可启动")
            return True
            
    except ImportError as e:
        print(f"❌ WatermarkTool 导入失败: {e}")
        print("   请检查项目结构和依赖安装")
        return False

def start_flask_app():
    """启动Flask应用"""
    print("\n🚀 启动Web服务...")
    print("=" * 60)
    print("🌐 服务地址: http://localhost:5000")
    print("📱 移动端访问: http://0.0.0.0:5000")
    print("🛑 停止服务: 按 Ctrl+C")
    print("=" * 60)
    
    try:
        # 启动Flask应用
        os.environ['FLASK_APP'] = 'app.py'
        os.environ['FLASK_ENV'] = 'development'
        
        # 直接运行app.py
        subprocess.run([sys.executable, 'app.py'])
        
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🎯 多模态水印工具 Web Demo 启动器")
    print("=" * 60)
    
    # 环境检查
    check_python_version()
    
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装依赖包")
        sys.exit(1)
    
    if not check_project_structure():
        print("\n❌ 项目结构检查失败")
        sys.exit(1)
    
    if not test_watermark_tool():
        print("\n❌ 水印工具测试失败")
        sys.exit(1)
    
    print("\n✅ 所有检查通过！")
    
    # 启动服务
    start_flask_app()

if __name__ == '__main__':
    main()