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


### `async`
- 并行：在多个cpu核心上，同时处理多个任务
- 并发：使用一个线程，看似同时处理多个任务

- I/O-Bound: 任务主要受IO操作（网络、硬盘读写）限制时，最适合使用 Async
- CPU-Bound：任务主要受CPU速度（数值计算）限制时，最适合使用多进程
- `asyncio`
  - Python标准库，导入 `async/await` 语法
- `Event Loop`
  - 管理 `coroutines` 的执行
  - 监控 `I/O` 事件
  - 基于当前什么任务已完成，什么任务在等待，决定接下来哪个 `coroutine` 先执行
  - 程序员通常不能直接与`Event Loop`交互
- `async def` 
  - 定义一个**可以暂停 可以后续恢复**执行的函数
  - 当调用一个`async def`函数时，通常不会立即运行，它会回传一个`coroutine object`，可以看作是一个计划或者一个可以运行的任务
  - 在函数体内，遇到 `await` 时，向`Event Loop`交出控制权
- `await`
  - 只可用在 `async def` 函数内部
  - 向`Python Event Loop` 交出控制权，类似于说，"我要等待当前任务运行，你可以先做其他任务"
- `asyncio.run()`
  - 用于最高层的 corutine 函数入口
  - 用于自己 APP 中的最高层
  - 不可在 `async def` 函数体内调用
- `asyncio.sleep(0)`
  - 在计算密集型任务中，如果想用 asyncio，则在任务开始前先使用 `asyncio.sleep(0)` 交出控制权
- `task_obj = asyncio.create_task(coroutine_obj)`
  - 将一个 `corutine obj` 包装为一个 `Task obj` 
  - 在 create_task return 后，即开始执行
  - 插入到 `Event Loop` 中
- `results = await asyncio.gather(task1, task2, task3)`
  - 等待 `Event Loop` 中的任务全部执行完成

### 并发、并行

- 并发 Concurrency
  - 多个任务看似同时进行
  - 可以使用多线程（multi-thread）或者协程（coroutine）实现
  - 协程 coroutine 通过 EventLoop 管理，暂停或者继续执行子程序 subroutine
  - 适用于 **I/O密集型** 任务
  - `concurrent.futures.ThreadPoolExecutor`  --- 多线程
  - `asyncio`  --- 协程

- 并行 Parallelism
  - 多个任务同时进行
  - 使用多进程（multi-processing）实现
  - 适用于 **CPU密集型** 任务
  - `multiprocessing.Pool`  --- 多进程

```python
import asyncio
import concurrent.futures
import multiprocessing
import concurrent
from concurrent.futures import ThreadPoolExecutor
import time

# Coroutine
async def fetch_data(num:int):
    print(f"Start fetch data: {num}")
    await asyncio.sleep(num)  # Simulating I/O operation
    print(f"Complete fetch data: {num}")
    return f"Data-{num}"

# Concurrent operation
def process_data(num):
    # I/O-bound operation
    print(f"Start Process Data: {num}")
    time.sleep(num)
    print(f"Finished Process Data: {num}")
    return f"process result: {num}"

# Parallel operation
def heavy_computation(num):
    # CPU-intensive operation
    print(f"Start Heavy Computation: {num}")
    time.sleep(num)
    print(f"Finished Heavy Computation: {num}")
    return f"heavy computation: {num}"

def on_result(result):
    print(f"Received Result: {result}")    

async def main():
    # Coroutine for I/O operations
    data = await asyncio.gather(
        asyncio.create_task(fetch_data(3)), 
        asyncio.create_task(fetch_data(2)), 
        asyncio.create_task(fetch_data(1)), 
        )
    for d in data:
        print(d)
    
    print("-------------------------")
    
    # Concurrent execution for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_data, i) for i in [3, 2, 1]]
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
        
    print("+++++++++++++++++++++++++++")
    
    # Parallel execution for CPU-bound tasks
    with multiprocessing.Pool(processes=3) as pool:
        # map 阻塞式
        results = pool.map(heavy_computation, [3, 2, 1])
        print(results)
        
        # apply_async 非阻塞式
        res = []
        for i in [3, 2, 1]:
            res.append(pool.apply_async(heavy_computation, args=(i,), callback=on_result))
            
        for r in res:
            r.wait()
            print(r.get())

if __name__ == '__main__':
    asyncio.run(main())
```


## 小案例



