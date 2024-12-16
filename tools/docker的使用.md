# Docker

## 安装
- 

## `Image` 管理
- `docker pull <image>` 从Docker Hub拉取指定镜像
- `docker images`  查看当前本地有的镜像
- `docker rmi <image>`  删除指定的镜像
- `docker run <options> <image>`  创建并启动一个容器
  - 每次执行 `docker run` 都会创建一个**新的容器实例**
  - `docker run -d`  以后台模式运行
  - `docker run -it`  以交互模式运行


## `Container` 管理

- `docker ps`  列出当前正在运行的容器
  - `docker ps -a`  列出所有容器，包括停止状态的
- `docker start <container>`  启动已经停止的容器
  - 继续使用已经存在的容器实例
- `docker stop <container_id>`  停止指定的容器
- `docker rm <container>`  删除已经停止的容器
- `docker exec -it <container> <command>`  在**运行的容器中**执行命令
  - `docker exec -it -u <user_name> <container> <command>`
- 退出docker运行环境 `ctrl+d` 

## 数据卷

- 为什么用数据卷：
  - 持久：数据卷能确保容器删除后数据不会丢失，即使容器停止或删除后，数据仍然保留在数据卷中；
  - 共享数据：多个容器共享一个数据卷，从而实现数据的共享和协作；
  - 提高IO性能：因为数据存在宿主机的文件系统中，而不是容器层中；
  - 简化备份和迁移：方便迁移
- 创建
  - `docker volume create my_volume`
- 使用
  - `docker run -d -v my_volume:/path/in/container my_image`
    - `-v my_volume:/path/in/container` 将数据卷挂载在容器指定路径下
- 查看
  - `docker volume ls`
- 删除
  - `docker volume rm my_volume`
- 使用本地目录作为数据卷
  - `docker run -d -v /path/in/host:/path/in/container my_image`

## 网络
- `Bridge`
  - 简介：Docker默认使用`bridge`模式来创建容器网络，每个docker容器都会连到默认的`bridge`网络上
  - 特点：在这种模式下，**容器之间默认是隔离的**，但可以通过配置端口映射来互相通信
  - 容器无法直接与宿主机或其他外部网络直接通信，除非进行端口映射到宿主机端口；
- `Host`
  - 简介：在`host`模式下，容器共享宿主机的网络命名空间，意味着容器可以直接访问宿主机的网络接口
  - 在这种模式下，容器就像直接在宿主机上运行的进程一样，可以直接访问外部网络而不需要特别的设置；
  - 但这也意味这容器之间的隔离性较低；
- `Overlay`
  - 简介：用于创建跨多个Docker宿主机的容器网络，它允许不同宿主机上的容器之间的通过容器网络进行通信；
  
## 使用`SSH`

## 案例

### 使用`PaddleX`

#### 拉取`Image`
- 在`docker desktop` 中可以搜索相应的项目镜像，选择相应的`Tag`进行`Pull`
- `docker pull paddlepaddle/paddle:3.0.0b1-gpu-cuda11.8-cudnn8.6-trt8.5`

#### 运行容器

- `docker run --gpus all --name paddlex --network=host -it <image_name> /bin/bash`
  - `--gpus all` 使用宿主机的GPU，在docker中可以使用`nvidia-smi`命令确认gpu是否可用
  - 测试`paddle`环境是否正确
    - `python -c "import paddle; paddle.utils.run_check()"`
    - `python -c "import paddle; print(paddle.__version__)"`
  - 获取源代码
    - `git clone https://gitee.com/paddlepaddle/PaddleX.git`
    - `pip install -e .`
    - `paddlex --install`  安装成功后会显示 "All packages are installed."
  - 测试模型效果
    - `paddlex --pipeline object_detection --model PP-YOLOE_plus-S --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/fall.png`
    - 
  
#### 使用 `SSH` 访问