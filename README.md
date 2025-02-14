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

      - 
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