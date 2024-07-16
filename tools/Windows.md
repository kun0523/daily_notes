# Windows

## 遇到过的问题

### 1. 可以远程但是忘记了登录密码
- 情景描述：在工作电脑上可以通过`mstsc`进行远程设备电脑，但是时间太久忘了设备电脑登录密码，windows上记录的密码不是明文，所以查不到密码
- 解决办法：
  - 控制面板
  - 用户账户 >> 管理账户 >> 更改密码

### 2. WSL 忘了用户密码
- windows中可以直接把`root`用户设置为默认用户，在`root`下修改其他用户密码
- `ubuntu1804 config --default-user root`  使用管理员开启`powershell`中运行
  - 不同的发行版，命令中的第一项需要进行相应的修改，例如使用的是`ubuntu 20.04`版本，就修改为`ubuntu2004 config --default-user root`
- 在`root`用户下修改指定用户的密码
  - `passwd user_name` : `user_name` 指定为你要修改的用户名，输入两遍新的密码即可修改完成
- 再次把默认用户修改为原来的用户