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

