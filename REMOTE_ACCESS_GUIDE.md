# 🌐 远程访问指南

## 📍 场景说明
代码在远程服务器上，需要在本地电脑的浏览器中访问Web界面。

## 🚀 方法一：SSH端口转发（推荐）

### 优点
- ✅ 安全性高，数据加密传输
- ✅ 不需要开放服务器防火墙端口
- ✅ 适用于任何网络环境

### 步骤

#### 1. 在服务器上启动Web服务
```bash
# 方法1：使用远程启动脚本
python start_server.py

# 方法2：使用原始启动脚本
python start_demo.py

# 方法3：直接启动
python app.py
```

#### 2. 在本地电脑新开终端，执行SSH端口转发
```bash
# 基本语法
ssh -L 本地端口:localhost:远程端口 用户名@服务器IP

# 实际示例
ssh -L 5000:localhost:5000 your_username@192.168.1.100

# 如果使用特定SSH端口
ssh -p 22 -L 5000:localhost:5000 your_username@192.168.1.100
```

#### 3. 在本地浏览器访问
打开浏览器，访问：**http://localhost:5000**

#### 4. 注意事项
- SSH连接必须保持不断开
- 如果SSH断开，需要重新建立连接
- 可以在后台运行：`ssh -f -N -L 5000:localhost:5000 user@server`

## 🔧 方法二：直接网络访问

### 优点
- ✅ 设置简单，无需SSH转发
- ✅ 多人可以同时访问

### 缺点
- ❌ 安全性较低
- ❌ 需要开放防火墙端口

### 步骤

#### 1. 在服务器上开放防火墙端口
```bash
# Ubuntu/Debian
sudo ufw allow 5000
sudo ufw reload

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload

# 或者临时关闭防火墙测试
sudo ufw disable
```

#### 2. 启动Web服务
```bash
python start_server.py
```

#### 3. 获取服务器IP地址
```bash
# 查看服务器IP
hostname -I
# 或
ip addr show
```

#### 4. 在本地浏览器访问
访问：**http://服务器IP:5000**

例如：`http://192.168.1.100:5000`

## 🐳 方法三：反向代理（高级）

如果需要通过域名访问或更复杂的部署：

### 使用Nginx反向代理
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🛠️ 故障排除

### 问题1：SSH端口转发无法连接
**解决方案：**
```bash
# 检查SSH服务状态
sudo systemctl status ssh

# 检查SSH配置
sudo nano /etc/ssh/sshd_config
# 确保 AllowTcpForwarding yes

# 重启SSH服务
sudo systemctl restart ssh
```

### 问题2：防火墙阻止访问
**解决方案：**
```bash
# 检查防火墙状态
sudo ufw status

# 检查端口占用
netstat -tlnp | grep 5000

# 临时开放端口测试
sudo ufw allow 5000
```

### 问题3：服务器IP不明确
**解决方案：**
```bash
# 查看所有网络接口
ip addr show

# 查看路由表
ip route show

# 使用外部服务查看公网IP
curl ifconfig.me
```

### 问题4：端口已被占用
**解决方案：**
```bash
# 查看端口占用
lsof -i :5000

# 杀死占用进程
sudo kill -9 PID

# 或者使用其他端口启动
# 修改app.py中的port=5001
```

## 📱 移动设备访问

如果要在手机/平板上访问：

1. **确保设备在同一网络**
2. **使用方法二（直接网络访问）**
3. **在移动浏览器输入：** `http://服务器IP:5000`

## 🔒 安全建议

1. **生产环境使用HTTPS**
2. **设置防火墙规则限制IP访问**
3. **使用VPN连接服务器网络**
4. **定期更换SSH密钥**

## 📋 快速检查清单

- [ ] 服务器上Web服务已启动
- [ ] 防火墙端口已开放（方法二）
- [ ] SSH端口转发已建立（方法一）
- [ ] 本地浏览器可以访问
- [ ] 网络连通性正常

## 🆘 获取帮助

如果仍然无法访问，请检查：

1. **服务器日志：** `tail -f watermark_demo.log`
2. **网络连接：** `ping 服务器IP`
3. **端口测试：** `telnet 服务器IP 5000`
4. **服务状态：** 查看终端输出信息

---

**推荐使用方法一（SSH端口转发），既安全又简单！** 🔐