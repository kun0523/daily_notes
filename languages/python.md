# Python

## Ubuntu使用Python

### 搭建虚拟环境
- 配置源
```sh
>> pip config get global.index-url  # 查看现在的源
>> pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

>> pip install openvino -i https://pypi.tuna.tsinghua.edu.cn/simple
>> pip install openvino -i https://mirrors.aliyun.com/pypi/simple
>> pip install openvino -i https://pypi.mirrors.ustc.edu.cn/simple
```

```sh
sudo apt update
sudo apt install python3 python3-venv
mkdir proj1
cd proj1
python3 -m venv env
source env/bin/activate
pip install packagename
deactivate

rm -rf env  // 删除指定的虚拟环境
```

- 使用conda 创建虚拟环境
```sh
conda create -n demo_env python=3.10
conda activate demo_env

conda create -p d:\envs\demo_env python=3.10
conda activate d:\envs\demo_env

pip install package_name --isolated  // 注意，有可能系统内已经有python翻译器，导致一些依赖项没有安装在指定的虚拟环境下，导致虚拟环境移植后运行报错，解决办法，将系统内已有的翻译器删除后，重新安装
pip install package_name -t path/to/install  // 指定安装位置

conda env remove -n demo_env --all  删除指定的虚拟环境

conda pack -n env_name -o path/to/save/file_name.tar.gz
conda pack -p env_path -o path/to/save/file_name.tar.gz
```

## 程序打包
### Pyside6-deploy
- `pyside6-deploy main.py --init`
  - 产生config文件
  - main.py 文件中包含整个APP的入口函数

- `pyside6-deploy -c pysidedeploy.spec`
  - 根据config文件进行打包

- 问题：
  - 使用skl2onnx 库，会出现打包失败，找不到源码的问题，未解决

### Pyinstaller
- `pyinstaller -F main.py -i icon.ico -n AppName -w` 
  - `-w`  屏蔽 cmd 黑窗
  - `-c`  保留黑窗


## 常用的三方库

### `sys`


```python
import sys

print(sys.version)  
print(sys.platform)

args = sys.argv()  # 获取命令行参数
if len(args) != 3:
    sys.stdout.write(f"[X] Error! {args[0]} needs Three args")  # 输出在终端
    sys.exit(1)  # 退出，并返回异常代码
username = args[1]
passwd = args[2]

command = sys.stdin.readline()  # 获取命令行输入

sys.exit(0)
```

### `request`

- 网络发包
  - 常用方法：Get/Post
  - Get方法传参：直接在URL中添加
  - Post方法传参：URL，data，files
  - 创建Session

```python
import requests

x = requests.get("http://httpbin.org/get")

print(x.headers)
print(x.headers["Server"])
print(x.status_code)

if x.status_code == 200:
    print("Success!")
elif x.status_code == 404:
    print("Not Found!")

print(x.elapsed)
print(x.cookies)
print(x.content)
print(x.text)

x = requests.get("http://httpbin.org/get", params={"id":1})
print(x.url)

x = requests.get("http://httpbin.org/get?id=2")
print(x.url)

x = requests.get("http://httpbin.org/get", params={'id':3}, headers={'Accept':"application/json", "tst_header":"test"})
print(x.text)

x = requests.delete("http://httpbin.org/delete")
print(x.text)

# post 方法可以上传的有： 1. params  2. data  3. files
x = requests.post("http://httpbin.org/post", params={'t':10}, data={"a":"b", 'c':'d'})
print(x.text)

# 使用post上传本地文件
files = {"txt_file":open('test.txt', 'rb')}
x = requests.post("http://httpbin.org/post", files=files)
print(x.text)

x = requests.get("http://httpbin.org/get", auth=("username","passwd"))
print(x.text)

'''
# 设置等待时间
x = requests.get("http://httpbin.org/get", timeout=0.01)
print(x.content)
'''

# 使用cookies
x = requests.Session()
x.cookies.update({'a':'b'})
print(x.get('http://httpbin.org/cookies').text)
print(x.get('http://httpbin.org/cookies').text)
print(x.get('http://httpbin.org/cookies').text)

# 怎么用session访问需要登录认证的网站？？

exit(0)
```

### pyQt
- IDE: Qt Creator
- 需要的库：
  - `pyqt5`
  - `pip install pyside6`
- 大致开发流程：
  - 在 `form.ui` 中设计界面，`pyside6-uic form.ui -o ui_form.py`  产生python代码（每次修改界面后，都需要执行该命令，重新生成python代码）
  - 在 `widget.py` 中编写业务逻辑
- 信号槽机制
- 界面画图

## 常用的语法特性

### `Decorators`

- 装饰器  特征：
  - 实质是一个函数，且接收另一个函数作为参数
  - 函数体内定义了一个函数，并作为返回值返回

```python
import time
import datetime

# 无参装饰器
def printInfo(func):
    def wrapper():
        print("-"*50)
        print(f"Call Func: {func.__name__}, at {datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')}")
        func()
        print(f"Complete at {datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')}")
        print("-"*50)
        print()
    return wrapper

@printInfo
def test():
    print("Do some task.....")
    time.sleep(1)


# 有参装饰器
def printInfoWithParams(func):
    def wrapper(*args, **kwargs):
        print("-"*50)
        print(f"Call Func: {func.__name__}, at {datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')}")
        func(*args, **kwargs)
        print(f"Complete at {datetime.datetime.today().strftime('%Y/%m/%d %H:%M:%S')}")
        print("-"*50)
        print()
    return wrapper

@printInfoWithParams
def test2(sleep_time):
    print("Do some tasks.....")
    time.sleep(sleep_time)


# 测试功能
test()
print("*"*50)
test2(2)
print("*"*50)
test2(3)
print("*"*50)
test2(4)
```

### `Generators`

- 生成器
  - 函数形式，或 Tuple表达式
  - 使用 `yield`  代替  `return`
  - 惰性返回，有需要时才返回一个值

```python
import time

# 函数形式
def genInt():
    n = 0
    time.sleep(0.5)
    n += 1
    yield n

    time.sleep(0.5)
    n += 1
    yield n

    time.sleep(0.5)
    n += 1
    yield n

res = genInt()
print(res)  # <generator object genInt at 0x000002AB24C902E0>
print(next(res))  # 1
print(next(res))  # 2

print("-"*50)
for i in genInt():
    print(i)

# Tuple 表达式
print("-"*50)
res2 = (i for i in range(5)) 
print(res2)  # <generator object <genexpr> at 0x000002AB24CBBBA0>
print(next(res2))  # 0
print(next(res2))  # 1
```

### `Serialization`
- 序列化


### `Closures`
- 闭包


### `Introduction`

### `Classes`

### `Encapsulation`
- 封装



## 小案例



