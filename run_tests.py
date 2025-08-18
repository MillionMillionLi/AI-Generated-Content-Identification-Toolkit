#!/usr/bin/env python3
"""
统一测试运行脚本
提供便利的测试入口点，自动设置正确的路径和环境
"""

import sys
import os
import subprocess
from pathlib import Path


def setup_environment():
    """设置测试环境"""
    
    # 确保从项目根目录运行
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    
    # 添加 src 目录到 Python 路径
    src_dir = script_dir / "src"
    if src_dir.exists():
        src_path = str(src_dir)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        os.environ['PYTHONPATH'] = f"{src_path}:{os.environ.get('PYTHONPATH', '')}"
        print(f"✅ 已设置 src 目录: {src_path}")
    else:
        print("❌ 错误: 找不到 src 目录")
        return False
    
    # 检查必要的依赖
    print("🔍 检查环境依赖...")
    
    required_modules = ['torch', 'transformers', 'PIL']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"  ❌ {module}")
    
    if missing_modules:
        print(f"\n⚠️ 缺少依赖模块: {', '.join(missing_modules)}")
        print("请先安装必要的依赖: pip install torch transformers pillow")
        return False
    
    return True


def run_test(test_file=None, verbose=False):
    """运行测试"""
    
    if not setup_environment():
        return False
    
    # 构建测试命令
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_file:
        if not test_file.startswith("tests/"):
            test_file = f"tests/{test_file}"
        cmd.append(test_file)
    else:
        cmd.append("tests/")
    
    print(f"\n🚀 运行测试命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        return False
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="统一测试运行脚本")
    parser.add_argument("test", nargs="?", help="要运行的测试文件 (可选)")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--audio", action="store_true", help="仅运行音频水印测试")
    parser.add_argument("--image", action="store_true", help="仅运行图像水印测试")
    parser.add_argument("--text", action="store_true", help="仅运行文本水印测试")
    parser.add_argument("--quick", action="store_true", help="快速测试 (跳过耗时测试)")
    
    args = parser.parse_args()
    
    # 根据参数确定测试文件
    test_file = args.test
    
    if args.audio:
        test_file = "test_audio_watermark.py"
    elif args.image:
        test_file = "test_image_watermark.py"
    elif args.text:
        test_file = "test_text_watermark.py"
    
    # 设置环境变量
    if args.quick:
        os.environ['QUICK_TEST'] = '1'
    
    print("🧪 统一水印工具测试运行器")
    print("=" * 60)
    
    success = run_test(test_file, args.verbose)
    
    if success:
        print("\n✅ 测试完成")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()