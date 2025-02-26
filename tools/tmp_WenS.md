# WS
  - 骨干网络：
    - ResNet、MobileNet、
    - Vit、Swin-Transformer
  - 目标检测
  - 关键点检测  Medipipe  动作识别
  - OCR
  - 部署：OpenVINO  TensorRT  量化

- 影响准确率的因素：
  - 数据：
    - 训练集与真实数据的偏差，样本不服从独立同分布
    - 做数据增强：
  - 模型结构：
    - 损失函数
  - 训练过程：
    - 过拟合与欠拟合
  - 部署：
    - 前处理差异

# Linux

## 用户权限管理
- `sudo adduser <username>`
- `sudo groupadd <groupname>`
- `sudo usermod -aG <groupname> <username>`  -aG 追加/添加用户到附加组
- `sudo usermod -g <groupname> <username>`  更改指定用户的所属组
- `id <username>`  查看用户的组信息
- `chmod <权限模式> <文件/目录路径>`
- 权限模式
  - `chmod 644 example.txt`
  - `chmod u+rw,go+r example.txt`
  - u 所有者  g 组  o 其他人
  - r 读   w 写  x 执行
  - 数字模式： `755` 所有者 rwx  组和其他人 r-x  使用三位二进制数
  - 符号模式：
- `sudo chown <new_owner>:<group> <filename>`

## 创建链接 `ln`

- 硬链接
  - 所有硬链接共享同一个文件内容，删除任一硬链接不会影响其他链接，**只有当最后一个硬链接删除时，文件数据才会被释放**；
  - `ln src_file.txt hard_link.txt`
  - 不可跨文件系统，硬链接必须与源文件在同一个磁盘分区
  - 不可链目录，只可以链文件
  - 可用于文件备份的场景
- 软连接
  - 软链接存储的是源文件的路径信息，**如果源文件被删除或移动，软链接将会失效**
  - `ln -s src_file.txt symbolic_link.txt`
  - 可跨文件系统，**可链接不同磁盘分区的文件或目录**
  - 用于，路径简化的场景

## `find`命令
- 按文件名查找  `find /path/to/directory -name file_name`
- 按文件类型查找  `find /path/to/directory -type f`  查找指定路径下所有的普通文件
- 按文件大小查找  `find /path/to/directory -size +100M` 查找指定路径下大于100MB的文件
- 按用户或组查找  `find /path/to/directory -user username`  查找指定目录下输入该用户的文件

# OpenCV

## resize的插值算法
- `INTER_NEAREST` 最近邻插值   直接选择距离目标像素最近的原始像素值，不进行任何平滑计算
- `INTER_LINEAR` 双线性插值  在目标像素周围的4个最近邻像素上进行线性加权插值  **默认推荐**
- `INTER_CUBIC`  双三次插值  使用16个邻近像素进行三次样条插值，通过多项式拟合平滑过渡
- `INTER_AREA`  区域插值  对原始区域内的像素取均值或加权平均  **适合图像缩小**

## 图像增强
- 增亮  ``
- 直方图均衡化  
  - `void cv::equalizeHist(InputArray src, OutputArray dst)`  输入 8bit 单通道图  输出归一化亮度提升对比度
- 仿射变换
  - `cv::Mat affineMat = cv::getAffineTransform(srcPoints, dstPoints);`  映射矩阵
  - `cv::Mat rotateMat = cv::getRotationMatrix2D(cv::Point2f(img.cols/2, img.rows/2)<旋转中心>,45<旋转角度（顺时针>, 0.8<缩放因子>);`  旋转矩阵
  - `void cv::warpAffine(InputArray	src, OutputArray dst, InputArray M, Size dsize, int	flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT,const Scalar& borderValue = Scalar())`
- 透视变换
  - `void cv::warpPerspective( InputArray src, OutputArray dst,InputArray M,Size dsize,int	flags = INTER_LINEAR,int borderMode = BORDER_CONSTANT,const Scalar&	borderValue = Scalar())`

## 二值化
- 简单阈值：
  - ``
- 大津法
  - 适用于灰度直方图呈现双峰的场景
- 三角法
  - 适用于灰度直方图呈现三角形的场景

## 形态学变换
 - 腐蚀/膨胀
 - 形态学梯度
 - 礼帽/黑帽
 - 击中击不中

## 边缘检测  ？？
- Sobel
  - 一阶微分，计算水平/垂直梯度
- Laplace
  - 二阶微分
- Canny
  - 多阶段优化（高斯滤波 -> 梯度计算 -> 非极大值抑制 -> 双阈值连接）

## 轮廓检测
- `findContours`
- 轮廓特征

## 模板匹配
- 单模板匹配
- 多尺度模板匹配（结合图像金字塔）

## 光流

## 傅里叶变换

# 骨干网络
- ResNet
- MobileNet
- Vit
  - 始终都是在整图上做自注意力机制，随着图片尺寸增大，注意力计算量是成指数增长的，不能用于大图；
- Swin-transformer
  - 在小窗口内算自注意力，只要窗口大小固定，则自注意力机制的计算量就固定，随着图片变大，计算量是线性增长的；（基于CNN的先验知识，视觉任务基于局部信息就可以）
  - 移动窗口，所以能够产生跨窗口的注意力
  - 有了多尺度的特征，所以才有利于处理下游任务
  - PatchMerging  类似pooling，从而获得多尺度的特征


# 目标检测

# 人脸识别
- 算法过程：
  - 使用人脸检测算法，抠出人脸
  - 使用CNN提取特征向量
  - Faiss做向量匹配，配对人

# 姿态识别

# 关键点检测

# 动作识别

# OCR
- 测试大模型对各种字体的识别能力，在线版和离线版

# 部署

## CPU-OpenVINO


## GPU-TensorRT


# LLM

## 基本原理

## 基本使用

## RAG使用

## 量化


# Qwen2.5-VL

- 下载HuggingFace上的模型
  - `huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ../qwen25vl/`  
- vllm
  - 用于推理大模型
  - 不能在windows上运行

## 量化
- AWQ Activation-aware Weight Quantization 

## 使用官方API

## 使用本地部署 - HuggingFace
- 

## 使用docker
- `docker pull nvidia/cuda:12.2.2-runtime-ubuntu22.04`  拉取相应的镜像
- `docker run --gpus=all -it --name vllm -p 8010:8000 -v D:\le_qwen\models:/llm-model  nvidia/cuda:12.2.2-runtime-ubuntu22.04`  启动容器
- `apt update -yq --fix-missing` 更新apt软件库， -y yes -q quiet 
- `apt install -yq --no-install-recommends pkg-config wget cmake curl git vim`
- `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- `sh Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3`
- `~/miniconda3/bin/conda init`
- `source ~/.bashrc`  安装好miniconda环境
- `conda create -n qwen python=3.11 -y`  qwen虚拟环境
- `pip install vllm`  使用vllm 下载模型并推理
- `pip install git+https://github.com/huggingface/transformers`
- `export HF_HOME=/llm-model`
- `vllm serve Qwen/Qwen2.5-VL-3B-Instruct --limit-mm-per-prompt image=4`

## 直接python加载模型推理
- huggingface-cli 下载模型
- 加载离线模型
```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


# 加载模型与处理器
model = AutoModelForImageTextToText.from_pretrained(
    "./models/",  # 指定模型路径
    torch_dtype=torch.bfloat16,  # 推荐使用 bfloat16 节省显存[4,5](@ref)
    device_map="auto"            # 自动分配 GPU/CPU 资源
)
processor = AutoProcessor.from_pretrained("./models/")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "./test_images/OIP-C.jpg"},
            {"type": "text", "text": "detect little girl in this picture, return boundingbox as {'obj':'girl', 'position':[top_left_x, top_left_y, bottom_right_x, bottom_right_y]}"}
        ]
    }
]

# 处理图像/视频输入
image_inputs, video_inputs = process_vision_info(messages)

# 生成模型输入格式
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

# 生成响应（推荐参数配置）
generated_ids = model.generate(
    **inputs,
    max_new_tokens=2048,          # 控制生成文本长度[4,7](@ref)
    do_sample=True,               # 启用随机采样
    top_p=0.9,                    # 核采样概率阈值
    temperature=0.7,              # 控制生成随机性
    repetition_penalty=1.1        # 减少重复生成
)

# 解码输出
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(output_text[0])
```

## 使用Ollama部署Qwen2.5-VL  视觉VL模型不支持 GGUF 转换！！！
- 下载模型文件
  - `huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir path/to/save`
- 转换成`GGUF`格式
  - 使用 `llama.cpp` 库 
    - `git clone https://github.com/ggml-org/llama.cpp.git`
    - `pip install -r requirements/requirements-convert_hf_to_gguf.txt`
    - `cmake -B build` 
    - `cmake --build build --config Release` 生成工具
    - 转换为 `gguf` 格式    
    - 模型量化  
- 创建 Ollama模型
- 推理测试

- 场景挖掘： 
- 算法工程落地  openvino  tensorrt  量化  

- python C++ code
- self-intro：视觉任务，数据增强辅助训练，模型优化，模型部署，大模型应用
- questions 
1. 模型结构的改动
   - 背景复杂，导致误检：通道注意力机制
   - 对象小：加大特征图分辨率
   - yolov10 注意力
   - yolov11 注意力
   - yolov12 注意力
   - 加小规模的注意力机制是趋势
   - 
2. 模型轻量化
   - 蒸馏（大模型标注数据，用于小模型训练）
   - ~~剪枝~~
   - 量化  (模型本身很小的时候量化反而变慢)
   - 
3. 部署工程化  yolo11s  100layers  20GFLOPs  （yolo11m  200layers   yolo11l 300layers）
   - python    cpu  640  det  yolo11s 18MB    120ms
     - 大致流程：
   - OpenVINO  cpu  640  det  yolo11s 36MB    80ms
     - 大致流程：
   - TensorRT  3060  640  det  yolo11s 22MB   10ms
     - 大致流程：反序列化模型创建engine >> 根据输入输出尺寸创建CUDA缓冲区域 >> 将数据同步到CUDA后进行推理 >> 将推理结果从CUDA中取出做后处理
   - 
4. 大模型使用
   - RTX 3090（24G） 部署 ollama deepseek R1 14b 配合 cherry studio 做本地知识库
   - Qwen2.5 7b ubuntu huggingface transformers 可以根据prompt做目标检测和OCR
   - RAG 知识库
     - nomic-embed-text
     - langchain 做文本抽取，组件向量

5. 常用指标理解：
   - Recall
   - Precision
   - mAP

6. ultralytics不同任务的训练方式：
   - 分类：
     - scales [depth(网络层数), width(中间特征图通道数), max_channels(限制最大的通道数)]
       - depth == e： 控制中间特征图的维度数
       - 

- 模块解释

- Conv
  - act(bn(conv2d))
  - 没有做残差连接

- Bottleneck
  - 两次卷积 + 恒等映射(如何卷积后的通道数==原始特征图的通道数)
  - 第二次卷积可以做分组卷积，即实现深度可分离卷积 (group_num = input_channel_num)

- SPP
  - 3个MaxPool堆叠（kernel=5,9,13，特征图尺寸不变，仅改变特征图感受野）  
  - 分别计算3个Kernel的MaxPooling
  - 相当于在通道维度上，拼接不同感受野的特征图，最后再做一次卷积，融合信息
  - 
- SPPF
  - 3个MaxPool堆叠，KernelSize只有一种 5，不改变特征图大小；
  - 依次计算3个k=5的MaxPooling，
  - 最后在通道维度上拼接，再做依次卷积

- CSP
  - Cross Stage Partial Net  跨阶段的局部网络
  - 对特征图，在通道维度上切分，分别进行不同的卷积网络，最后做特征融合
  - torch.tensor.chunk(chunks_num, dim)  对tensor的指定维度上，切分成 chunks_num 份  


- Attention
  - 进入之前已经压缩成  BxNx1x1 的形状   前面已经压成一维了吗？？？
  - qkv  1x1 卷积  >> B x HeadNum x （k_dim+q_dim+head_dim） x N
  - attn = softmax(q@k*scale)   ???? 为什么最后都变成一维了？？对head的权重？？待确认
  - V@attn

- PSABlock
  - Attention模块提取特征
  - FFN feed-forward 融合特征
  - 

- A2C2f
  - Area-Attention  ?????
  - 

- SE 