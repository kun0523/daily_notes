# Docker

## 安装

### Ubuntu 环境下

1. 添加 gpg key
   ```bash
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
   ```
2. 下载 deb 文件
   - `https://desktop.docker.com/linux/main/amd64/docker-desktop-amd64.deb?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-linux-amd64`
3. 安装
   ```bash
    sudo apt-get update
    sudo apt-get install ./docker-desktop-amd64.deb
   ```
   
4. 启动 daemon 
   - 上一步完成后，在所有应用中应该可以看到 Docker Desktop 应用
   - 运行 `Docker Desktop`
   - 如果想要向DockerHub上传镜像，则需要登录账号，如果只是拉取镜像，则不需要登录账户

5. 配置开启自启
   - `sudo systemctl enable docker`

6. 更换镜像源
   1. 如果在使用`docker pull`时报错:`docker: Error response from daemon: Get "https://registry-1.docker.io/v2/": context deadline exceeded` 可能是访问镜像源的问题，可以更换国内源
   2. ```json
      {
        "registry-mirrors" : ["https://docker.registry.cyou",
          "https://docker-cf.registry.cyou",
          "https://dockercf.jsdelivr.fyi",
          "https://docker.jsdelivr.fyi",
          "https://dockertest.jsdelivr.fyi",
          "https://mirror.aliyuncs.com",
          "https://dockerproxy.com",
          "https://mirror.baidubce.com",
          "https://docker.m.daocloud.io",
          "https://docker.nju.edu.cn",
          "https://docker.mirrors.sjtug.sjtu.edu.cn",
          "https://docker.mirrors.ustc.edu.cn",
          "https://mirror.iscas.ac.cn",
          "https://docker.rainbond.cc",
          "https://do.nark.eu.org",
          "https://dc.j8.work",
          "https://dockerproxy.com",
          "https://gst6rzl9.mirror.aliyuncs.com",
          "https://registry.docker-cn.com",
          "http://hub-mirror.c.163.com",
          "http://mirrors.ustc.edu.cn/",
          "https://mirrors.tuna.tsinghua.edu.cn/",
          "http://mirrors.sohu.com/" 
      ],
        "insecure-registries" : [
        "registry.docker-cn.com",
        "docker.mirrors.ustc.edu.cn"
        ],
        "debug": true,
        "experimental": false,
        "log-driver": "json-file",
        "max-concurrent-downloads":10,
        "max-concurrent-uploads":5,
        "runtimes": {
            "nvidia": {
                "args": [],
                "path": "nvidia-container-runtime"
            }
        }
      }
      ```

## `Image` 管理
- `Image` 镜像，相当于一个运行环境的快照，可以把自己调试好的环境，推送到DockerHub，在其他地方拉取后直接使用；
- `docker pull <image>` 从Docker Hub拉取指定镜像
- `docker images`  查看当前本地有的镜像
- `docker rmi <image>`  删除指定的镜像
- `docker image rm <image_id>` 删除指定的镜像
- `docker run <options> <image>`  创建并启动一个容器
  - 每次执行 `docker run` 都会创建一个**新的容器实例**
  - `docker run -d`  以后台模式运行
  - `docker run -it`  以交互模式运行
- `docker search <image_name>` 从DockerHub中搜索指定的镜像

### 使用`dockerfile`创建image
- 一个简单dockerfile：
```dockerfile
FROM ubuntu:22.04
CMD ["echo", "Hello From my first docker demo"]
```
- `docker build -t <image_name> .`
- `docker tag <curr_image_name> <docker_hub_username>/<new_image_name>:<tag_id>`

### 推送到DockerHub
- 通过代理访问DockerHub
  - `vim /etc/docker/daemon.json`  新增proxy 配置
  ```json
    "proxies":{
	    "http-proxy": "http://proxy_server:port",
	    "https-proxy": "http://proxy_server:port"
    }
  ```
- `docker-credential-pass`创建公钥
  - 下载：`https://github.com/docker/docker-credential-helpers/releases/download/v0.6.0/docker-credential-pass-v0.6.0-amd64.tar.gz`
  - `tar -xf docker-credential-pass-v0.6.0-amd64.tar.gz`
  - `chmod u+x docker-credential-pass`
  - `mv docker-credential-pass /usr/bin/`  方便命令调用
  - `gpg --generate-key`  用来产生一对新的GPG密钥对
  - `pass init gpg_id`  
    - `pass` 是密码管理器
- 在DockerHub上创建密码token
- `docker login -u dockerhub_username`
- `docker tag <curr_image_name> <dockerhub_username>/<image_name>:<tag_id>`
- `docker push <dockerhub_username>/<image_name>:<tag_id>` 就可以在DockerHub 自己的仓库中看到上传的镜像
- `docker pull <dockerhub_username>/<image_name>:<tag_id>` 拉取上传的镜像文件


## `Container` 管理
![容器的生命周期](../image_resources/container_lifecycle.png)
- `docker run -d -p 8080:80 --name web-server nginx`
  - 根据镜像的名称，启动容器
  - 不解析镜像名称后的参数，所以参数要写在镜像名称之前
  - `-d`  detach
  - `-p`  HostPort:ServerPort 本地端口：容器中服务的端口
  - `--name` ContainerName
  - `docker run` = `docker create` + `docker start`

- `docker ps`  列出当前正在运行的容器
  - `docker ps -a`  列出所有容器，包括停止状态的

- 停止容器：
  - `docker stop 容器名称` 通过发送stop信号，来停止容器，资源回收更好
  - `docker kill 容器名称` 直接关闭进程，如果内存中还有数据，则直接丢失

- `docker start <container>`  启动已经停止的容器
  - 继续使用已经存在的容器实例
- `docker stop <container_id>`  停止指定的容器
- `docker rm <container>`  删除已经停止的容器
- `docker exec -it <container> <command>`  在**运行的容器中**执行命令
  - `docker exec -it -u <user_name> <container> <command>`
- 查看在运行的容器的id：`docker ps -q`
- 停止所有在运行的容器：`docker stop $(docker ps -q)`
- 删除所有容器：`docker rm $(docker ps -aq)`
- 退出docker运行环境 `ctrl+d` 
- `docker stats` 查看各个容器对资源的使用情况

## 环境变量

- 在dockerfile中可以直接设定环境变量：`ENV PORT=8080`
- 可以在创建容器时，传入环境变量值：`docker run -e PORT=3000`
- 也可以在 `.env` 文件中设定环境变量值，并在创建容器时指定环境变量文件：`docker run --env-file .env`
- 注意！`.env` 往往环境变量文件中常常包含隐私信息，所以要避免打包到镜像中！！！要在`.dockerignore`中进行屏蔽：`**/.env*`

## 数据卷

- 本地挂载卷  Bind Volume：将主机本地的路径映射到容器内
- 命名卷 Named Volume：优势是，可以多个容器绑定同一个Volume，当Volume中内容发送改变时，多个容器同时改变

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
- `docker network ls` 查看运行中的容器的网络类型
- `Bridge`
  - 简介：Docker默认使用`bridge`模式来创建容器网络，每个docker容器都会连到默认的`bridge`网络上
  - 特点：在这种模式下，**容器之间默认是隔离的**，但可以通过配置端口映射来互相通信
  - 容器无法直接与宿主机或其他外部网络直接通信，除非进行端口映射到宿主机端口；
- `Host`
  - 简介：在`host`模式下，容器共享宿主机的网络命名空间，意味着容器可以直接访问宿主机的网络接口
  - 在这种模式下，容器就像直接在宿主机上运行的进程一样，可以直接访问外部网络而不需要特别的设置；
  - 但这也意味这容器之间的隔离性较低；
  - 当主机有使用代理时，使用该模式下，容器中可以直接使用主机的代理服务器
- `Overlay`
  - 简介：用于创建跨多个Docker宿主机的容器网络，它允许不同宿主机上的容器之间的通过容器网络进行通信；

- `docker build` 时，因本地的代理导致的库安装失败：
  - `docker build --network=host -t <image_name>:v0.1 .`

- 在容器中使用`apt-get update` 报错：`Failed to fetch http://deb.debian.org/debian/dists/bookworm-updates/InRelease  Unable to connect to 127.0.0.1:7899`
  - 代理设置错误，继承了主机的代理设置，需要把代理设置清空
  - `export http_proxy=`
  - `export https_proxy=`
  - 将代理清空后即可正常进行
  
## 使用`SSH`

## Docker Compose

- 可以同时管理多个docker contiainer
- 查看compose版本：`docker compose version`
- 完整生命周期操作：
  - 启动：`docker compose up -d`
  - 查看运行状态：`docker compose ps`
  - 查看日志（调试，**仅在运行过程中可以查看**）：`docker compose logs`
  - 停止(保留数据卷，删除容器)：`docker compose down`
  - 停止并删除数据卷和镜像：`docker compose down -v --rmi all`
  - 清理残留的资源：`docker system prune -a`
- 

## 镜像配置加速

- `daemon.json` 文件位置： `/etc/docker/daemon.json`
```json
 {
    "registry-mirrors": [
      "https://docker.xuanyuan.me",
      "https://docker-0.unsee.tech",
      "https://docker.hlmirror.com",
      "https://docker.imgdb.de",
      "https://docker.m.daocloud.io",
      "https://mirror.ccs.tencentyun.com"
    ],
    "log-driver": "json-file",
    "log-opts":{
      "max-size": "100m"  // 日志文件最大100MB，超过后拆分文件
      },
    "max-concurrent-downloads":10,
    "max-concurrent-uploads":5
  }
```
- 修改完后要重启 docker 服务
  - `sudo systemctl daemon-reload`
  - `sudo systemctl restart docker`
  - `sudo systemctl status docker`  查看docker状态  判断是否在运行

## Docker优化

### 优化镜像大小
- `docker history <image_name>`  可以查看创建该image的过程，以及每一步镜像文件的大小
- 选取影响最大的几步进行优化

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