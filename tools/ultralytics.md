# Ultralytics 框架使用

## 环境搭建


## 分类任务


## 检测任务

### 图片前处理
1. LetterBox Resize
2. BGR --> RGB (im[..., ::-1])   BHWC --> BCHW (im.transpose((0,3,1,2)))
3. im /= 255.0

- `yolo predict task=classify model=best.onnx source=D:\share_dir\iqc_crack\ultr_workdir\crack_cls\0110_yolo11s_sgd_lr00052\weights\ng device=cpu imgsz=224`


## 分割任务