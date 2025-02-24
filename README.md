# daily_notes
take notes about coding

# Goals 2025
1. cpp  - done
2. cmake  - 
3. linux  - 
4. design pattern  - part1/02/01
    - momento design pattern  备忘录设计模式
5. ocr
    - Intelligently extract text 05

- 编程语言
  - cpp + cmake
  - 设计模式  0/32
  - 编程题目
- OpenCV
  - 基础视觉算法应用
- 深度学习算法
  - 基础模块
    - [梗直哥 深度学习必修课：进击算法工程师 48/100](https://pan.baidu.com/disk/main?from=homeFlow#/index?category=all&path=%2F%E6%A2%97%E7%9B%B4%E5%93%A5%E2%80%93%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BF%85%E4%BF%AE%E8%AF%BE%EF%BC%9A%E8%BF%9B%E5%87%BB%E7%AE%97%E6%B3%95%E5%B7%A5%E7%A8%8B%E5%B8%88)
  - YOLO  
    - [唐宇迪课程 06 yolov1~v11](https://www.bilibili.com/video/BV1JFroYvEF4?spm_id_from=333.788.player.switch&vd_source=780f0c38dd8ea3940e5658b8ec24ea38&p=8)
  - OCR
    - paddleocr
    - RapidOcr
  - StableDiffusion
- LLM
  - 大模型的使用
  - Agent
  - RAG
  - DeepSeek 算法思想

- 0212
  - ~~PaddleX  LCNet  MobileNet~~
  - ~~FastDeploy  cls  OpenVINO~~
  
- 0213
  - ~~RapidOCR~~
  - PaddleX OCR
  - ~~FastDeploy OCR~~
  - TensorRT 配置
  - 机器学习在工业上的应用 预测性的维护，智能排程？

- 集成算法：
  - Boosting
    - AdaBoost (Adaptive Boosting): 调整样本权重和弱分类器的权重（迭代过程中加大错分样本权重，减小犯错率高的分类器权重），逐步聚焦难分类样本，最终加权组合
    - GBDT (Gradient Boosting)：拟合前一个分类器的残差
  - Bagging  Bootstrap Aggregating
    - RandomForest: 每个弱分类器是基于部分样本和部分特征进行训练的
- 决策树
  - 分叉的指标：信息增益（ID3），信息增益率（C4.5），基尼指数（CART），

- 项目过程：
  - 组织小样本集，基于预训练模型使用默认参数，快速实验，判断是否Work
  - 确认算法可行后，使用训练得到的模型，标注更多样本，从BadCase中分析当前模型的问题，对哪种场景识别较差
  - 人工修正标注后，重新训练模型，调整超参数，重复几次后可以得到较好的识别精度
  - 再考虑模型推理速度，如果不满足速度要求，尝试由大模型蒸馏出轻量化小模型，或者使用小模型基于目前的数据集训练


- yolo
  - yolov5
    - 算法细节
      - ![网络结构](./image_resources/yolov5_arch.png)
      - C3 / CSPLayer  3个卷积 + n个BottleNeck
      - SPPF Spatial Pyramid Pooling - Fast：
        - 使用 多个MaxPooling（k=5），产生不同尺度的特征图
        - MaxPooling通过padding的方式保持特征图大小不变
        - 最后concate，并通过1x1卷积恢复通道数量
        - 用于增大特征图感受野，从而实现不同尺度特征融合
      - 
  - yolov8
    - 算法细节
      - ![网络结构](./image_resources/yolov8_arch.png)
      - Anchor Free: 解耦头 Decoupled Head 设计 将分类任务和回归任务分离，使用独立分支处理不同任务
      - C2f：包含一个Split和多个BottleNect（就是2层3x3卷积），将特征图沿channel分成两部分，一部分走BottleNect，另一部分直接连输出，两部分拼接后输出

  
  - yolo11
    - 算法细节：
      - C3k2 继承于 C2f 仅改变 BottleNeck的方式
      - C3k  继承于 C3  
      - C2PSA  加入注意力机制
      - 
  - 部署：
    - openvino
      - 直接推理
      - 量化推理
    - TensorRT
      - 直接推理
        - 模型文件需要转成engine
      - 量化推理

- ppocr
  - 网络：
    - dbnet
    - crnn
    - KIE
  - 微调
    - 
  - 推理
    - OpenVINO
    - TensorRT


- 0215
  - tensorrt cls det **seg**
  - tensorrt **linux**
  - **dbnet crnn kie**
  - ppocr 微调
  - openvino

- KIE
  - key information extraction 从文本或图像中提取结构化关键信息的技术
  - SER 语义实体识别：对文本进行分类标注 
  - RE  关系抽取：建立实体间的关联

- 大模型 LLM
  

- TensorRT  rtx 2060 cls   det 20ms  seg  推理 + 延时统计 （1h）
  - rtx 2060 yolo11n engine fp16
    - cls  4.2 GFlops/112 layers/ 9.5 MB / 7ms
    - det  6.5 GFlops/238 layers / 7.8 MB / 25 ms
    - seg  10.4 GFlops/265 layers/ 8.6 MB / 
  - rtx 3060 yolo11s engine fp16
    - cls   GFlops/112 layers/  MB / ms
    - det   GFlops/238 layers / MB / ms
    - seg   GFlops/265 layers/ MB / 25ms
    - 
- OpenVINO  cls  det  seg  推理 + 延时统计 （1h）
  - i7-10 yolo11n onnx fp16
    - cls  4.2 GFlops/112 layers/ 10 MB / 10ms
    - det  6.5 GFlops/238 layers / 10 MB / 60ms
    - seg  10.4 GFlops/265 layers/ 11 MB / 
  - 
- TensorRT 量化
- OCR
  - DBNet
  - CRNN + CTC Loss
  - fastdeploy ocr fps ... TODO...
- OpenCV 主要算法
  - 对比度增强  平衡直方图
  - 直方图
  - Contual 轮廓
  - 二值化

- 需要熟练掌握的：
  - 模型训练:cls  det(track)  seg  pose  
    - YOLO 完整的训练过程  图像前处理
    - ViT  Swin-Transformer
  - 模型部署:ONNX OpenVINO TensorRT
    - OpenVINO 量化
    - TensorRT 量化
  - OCR微调
    - 训练过程
    - OCR是怎么识别出一个序列的？？
  - OpenCV传统算法
    - 
  - Qwen2.5-vl  做 图片理解 ocr  目标检测  
    - 怎么做LoRA微调！！！

- 数据、流量、场景、数据分析
- 解决问题，分析数据，使用大模型训练行业垂类模型


- 项目中主要困难点：
  1. 异常数据占比低
  2. 小目标检测
  3. 密集目标检测
  4. 对象跟踪
  5. 动作识别

- 问题：
  1. 对象跟踪，是不是只能基于目标检测？
  2. 密集目标怎么做跟踪
  3. 动作识别的原理
  4. huggingface的使用 diffuser tranformers pytorch-models
  5. P2PNet 密集计数

1. 复现经典骨干网络 -- 熟悉经典网络的数据流、pytorch搭建过程
2. 复现经典下游任务 目标检测  计数  跟踪
3. 大模型算法思想
4. 大模型微调
5. 视觉大模型应用

- TODO：
  - 手搭 Mobile net ！！！
  - 手搭 Transformer
  - 手搭 Vit  ！！！
  - 手搭 Swin-Transformer
  - 复现 P2PNet
  - LoRA微调

- 0220
  - **填表格**
  - ViT
  - Swin-Transformer
  - MobileNet V3 V4
  - yolo
  - Qwen-VL-7b-instructed for dent det
  - cherrystudio + ollama



