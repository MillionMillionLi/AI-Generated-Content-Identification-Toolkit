#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
远程服务器启动脚本
显示本地访问的详细信息
"""

import os
import sys
import socket
import subprocess
from pathlib import Path

def get_server_ip():
    """获取服务器IP地址"""
    try:
        # 尝试获取外网IP
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            # 过滤掉回环地址
            external_ips = [ip for ip in ips if not ip.startswith('127.') and not ip.startswith('172.17.')]
            if external_ips:
                return external_ips[0]
    except:
        pass
    
    # 备用方法
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "未知"

def check_port_availability(port=5000):
    """检查端口是否可用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result != 0  # 如果连接失败，说明端口可用
    except:
        return True

def show_access_info():
    """显示访问信息"""
    server_ip = get_server_ip()
    port = 5000
    
    print("🌐 远程访问信息")
    print("=" * 60)
    print(f"📍 服务器IP地址: {server_ip}")
    print(f"🚪 服务端口: {port}")
    print()
    
    print("🔗 本地访问方式:")
    print("1. SSH端口转发 (推荐):")
    print(f"   在本地执行: ssh -L {port}:localhost:{port} your_username@{server_ip}")
    print(f"   然后访问: http://localhost:{port}")
    print()
    
    print("2. 直接网络访问:")
    print(f"   浏览器访问: http://{server_ip}:{port}")
    print("   (需要防火墙开放端口)")
    print()
    
    print("🛡️ 防火墙配置:")
    print(f"   sudo ufw allow {port}")
    print("   sudo ufw reload")
    print()
    
    print("📋 SSH端口转发详细步骤:")
    print("   1. 在本地终端执行SSH转发命令")
    print("   2. 保持SSH连接不断开")
    print("   3. 在本地浏览器访问 localhost:5000")
    print("   4. 使用完毕后可关闭SSH连接")
    print()
    
    print("🔧 如果无法访问，请检查:")
    print("   - 防火墙设置")
    print("   - 网络连通性")
    print("   - SSH服务状态")
    print("=" * 60)

def main():
    """主函数"""
    print("🚀 多模态水印工具 - 远程服务器启动")
    
    # 显示访问信息
    show_access_info()
    
    # 检查端口
    if not check_port_availability():
        print("⚠️ 端口5000已被占用，请检查是否有其他服务在运行")
        print("   可以使用: lsof -i :5000 查看占用进程")
        return
    
    # 启动确认
    try:
        input("\n按回车键启动Web服务器...")
    except KeyboardInterrupt:
        print("\n启动已取消")
        return
    
    # 启动服务
    print("\n🌟 启动Web服务器...")
    print("服务启动后，请在本地使用上述方法访问")
    print("按 Ctrl+C 停止服务")
    print("-" * 40)
    
    # 启动Flask应用
    try:
        from app import app, init_watermark_tool
        
        # 初始化
        if not init_watermark_tool():
            print("❌ 水印工具初始化失败")
            return
        
        # 启动服务
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # 生产模式
            threaded=True
        )
        
    except ImportError:
        print("❌ 无法导入app模块，请确保在正确的目录运行")
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == '__main__':
    main()